"""
Indexed jump tables that do not start at their displacement.

Everything else in the toolkit reads `jmp dword ptr [reg*4 + disp]` as "a
table of code pointers starting at disp", and scans forward from disp until
a value stops looking like a code address. That is right for an ordinary
switch, whose index runs 0..N. It is wrong when the index is negative, and
MSVC's hand-written CRT memcpy/memmove (memcpy.asm) does exactly that in two
places:

    CopyDown:   ...  shr ecx, 2 ; cmp ecx, 8 ; jb CopyUnwindDown
    CopyUnwindDown:
                neg  ecx                                  ; ecx = -7 .. 0
                jmp  dword ptr [ecx*4 + UnwindDownVec+28]

    CopyLeadUp: ...  sub ecx, 4 ; jb ByteCopyUp
    ByteCopyUp:
                jmp  dword ptr [ecx*4 + TrailUpVec+16]    ; ecx = -4 .. -1

In both, disp names the table's *last* entry or one past its end. A forward
scan starting there reads code, finds at most one entry, and the rest of the
table -- and every case body only it points to -- is lost. BLACK's memmove
(0x000A24E0) hits the first shape at 0x000A26A6: the table is 0x000A275C..
0x000A277B, the forward scan saw only its last entry, the disassembler ended
the function at 0x000A2782 and the lifter sent the dispatch through
RECOMP_ITAIL, which cannot resolve an internal label such as 0x000A2784. The
copy then silently skipped its dword moves.

The same file biases a table the other way when the index is known to be at
least 1. Copying to an unaligned destination masks the misalignment and
indexes a three-entry table with it:

    CopyLeadUp: mov eax, edi ; ... ; and eax, 3 ; add ecx, eax
                jmp  dword ptr [eax*4 + LeadUpVec-4]      ; eax = 1 .. 3

(`test edi, 3; jnz CopyLeadUp` is what rules out 0.) The slot at the
displacement is the tail of the preceding instruction, so the forward readers
find nothing at all. BLACK's memmove has this at 0x000A252D (LeadUpVec
0x000A2544) and 0x000A26C1 (LeadDownVec 0x000A26D0): every copy of four
bytes or more to an unaligned destination fell through RECOMP_ITAIL.

This module proves the index range of such a dispatch from the instructions
that feed it, and only then says where the table is. It is deliberately
narrow:

  * The dispatch must be `jmp dword ptr [reg*4 + disp]` -- scale 4, no base.
  * Inside the dispatch's own basic block the index may only pass through
    `neg reg`, or be masked by `and reg, N`, which alone proves 0..N; any
    other write to it, or a call, abandons the analysis.
  * Without a mask, every edge into that block must bound the index with
    the instruction immediately before a conditional branch:
        cmp reg, N ; jb/jbe   (taken)      or   jae/ja (fall-through)
        sub reg, K ; jb       (taken)      or   jae    (fall-through)
    An unconditional jump in, a straight-line fall-through, or any branch
    whose flags do not come from that one instruction makes the range
    unknown.
  * The caller only acts where the forward reader is provably wrong: a
    range whose low end is negative, or a displacement slot that does not
    hold a code pointer. Every ordinary switch keeps the forward reader.
  * Slots at either end of the proven range that do not hold a code pointer
    are index values something outside this analysis rules out (the `test
    edi, 3` above); they are left out of the table rather than claimed as
    data. A hole inside the range means the proof and the bytes disagree,
    and nothing is claimed.

Nothing is guessed from pointer shapes: the entry count comes from the guard,
not from how far plausible-looking values happen to extend. Pointer checks
can only shrink a proven table, never grow one.

The analysis is shared by the disassembler (tools/disasm/engine.py) and the
code generator (tools/recomp/translator.py), which see different instruction
objects. Both expose `address`, `size`, `end_address`, `mnemonic`, `op_str`,
`jump_target`, `is_jump`, `is_cond_jump`, `is_ret` and `is_call`, and that is
all this module reads.
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Largest table a guard may prove. A copy loop's unwind table has 8 entries;
# this only rejects a nonsense bound, it does not size anything.
MAX_ENTRIES = 256

_FAMILIES = {
    "eax": ("eax", "ax", "al", "ah"),
    "ebx": ("ebx", "bx", "bl", "bh"),
    "ecx": ("ecx", "cx", "cl", "ch"),
    "edx": ("edx", "dx", "dl", "dh"),
    "esi": ("esi", "si"),
    "edi": ("edi", "di"),
    "ebp": ("ebp", "bp"),
    "esp": ("esp", "sp"),
}
_FAMILY_OF = {alias: reg for reg, aliases in _FAMILIES.items()
              for alias in aliases}

# Instructions that write general registers without naming them as the first
# operand. Treated as clobbering every register: this list only has to be
# conservative, never precise.
_IMPLICIT_CLOBBER = frozenset({
    "call", "cdq", "cwd", "cwde", "cbw", "mul", "div", "idiv",
    "cpuid", "rdtsc", "rdtscp", "rdmsr", "rdpmc", "xgetbv",
    "lodsb", "lodsw", "lodsd", "scasb", "scasw", "scasd",
    "movsb", "movsw", "movsd", "stosb", "stosw", "stosd",
    "cmpsb", "cmpsw", "cmpsd", "insb", "insw", "insd",
    "outsb", "outsw", "outsd", "in", "loop", "loope", "loopne",
    "loopz", "loopnz", "popal", "popad", "int", "int3", "into",
    "syscall", "sysenter", "enter", "leave", "xlatb", "lahf",
    "aaa", "aas", "aad", "aam", "daa", "das", "cmpxchg8b",
})
# First operand is read, not written.
_READ_ONLY_FIRST = frozenset({"cmp", "test", "push", "bt"})
# Both operands are written.
_WRITE_BOTH = frozenset({"xchg", "xadd", "cmpxchg"})

_JB = frozenset({"jb", "jc", "jnae"})
_JBE = frozenset({"jbe", "jna"})
_JAE = frozenset({"jae", "jnb", "jnc"})
_JA = frozenset({"ja", "jnbe"})

_DISPATCH_RE = re.compile(
    r"^dword ptr \[(?P<reg>eax|ebx|ecx|edx|esi|edi|ebp)\*4 \+ "
    r"(?P<disp>0x[0-9a-f]+|\d+)\]$")
_LEA_PAD_RE = re.compile(r"^(?P<r>\w+), \[(?P=r)\]$")


@dataclass
class JumpTableSpan:
    """Where a proven-index dispatch's table lives, and what it holds."""
    dispatch: int                 # address of the jmp
    disp: int                     # operand displacement (not the table start)
    lo: int                       # index range, inclusive: the proven range
    hi: int                       # less end slots that hold no code pointer
    slots: List[Tuple[int, int]]  # (slot VA, code pointer) for lo..hi

    @property
    def start(self) -> int:
        return (self.disp + 4 * self.lo) & 0xFFFFFFFF

    @property
    def end(self) -> int:
        return (self.disp + 4 * self.hi + 4) & 0xFFFFFFFF

    @property
    def targets(self) -> List[int]:
        """The code pointers, in table (address) order."""
        return [v for _, v in self.slots]


def parse_indexed_dispatch(insn) -> Optional[Tuple[str, int]]:
    """(index register, displacement) of `jmp dword ptr [reg*4 + disp]`."""
    if insn.mnemonic != "jmp" or insn.jump_target is not None:
        return None
    m = _DISPATCH_RE.match(insn.op_str.strip())
    if not m:
        return None
    return m.group("reg"), int(m.group("disp"), 0)


def _split_operands(op_str: str) -> List[str]:
    parts, depth, cur = [], 0, []
    for ch in op_str:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur or parts:
        parts.append("".join(cur).strip())
    return parts


def _reg_imm(insn) -> Optional[Tuple[str, int]]:
    ops = _split_operands(insn.op_str)
    if len(ops) != 2 or ops[0] not in _FAMILIES:
        return None
    try:
        return ops[0], int(ops[1], 0)
    except ValueError:
        return None


def _base_mnemonic(insn) -> str:
    m = insn.mnemonic.lower()
    if m.startswith("lock "):
        m = m[5:]
    return m


def _writes(insn, reg: str) -> bool:
    """Might `insn` change `reg` (or any part of it)? Errs toward yes."""
    m = _base_mnemonic(insn)
    if m.startswith("rep") or m in _IMPLICIT_CLOBBER:
        return True
    if m == "imul" and "," not in insn.op_str:
        return True                      # one-operand form writes edx:eax
    if m == "pop" and insn.op_str.strip() in ("", "esp"):
        return True
    if m in _READ_ONLY_FIRST or insn.is_jump or insn.is_cond_jump:
        return False
    ops = _split_operands(insn.op_str)
    dests = ops if m in _WRITE_BOTH else ops[:1]
    return any(_FAMILY_OF.get(d) == reg for d in dests if "[" not in d)


def _is_padding(insn) -> bool:
    """Alignment filler MSVC places between a terminator and the next label."""
    m = _base_mnemonic(insn)
    if m in ("nop", "int3"):
        return True
    if m == "mov":
        ops = _split_operands(insn.op_str)
        return len(ops) == 2 and ops[0] == ops[1] and ops[0] in _FAMILIES
    if m == "lea":
        return bool(_LEA_PAD_RE.match(insn.op_str.strip()))
    return False


class CodeView:
    """Address-indexed access to a set of decoded instructions."""

    def __init__(self, instructions: Iterable):
        self._by_end: Dict[int, List] = {}
        self._branches_to: Dict[int, List] = {}
        for insn in instructions:
            self._by_end.setdefault(insn.end_address, []).append(insn)
            if ((insn.is_jump or insn.is_cond_jump)
                    and insn.jump_target is not None):
                self._branches_to.setdefault(insn.jump_target, []).append(insn)

    def ending_at(self, addr: int) -> List:
        return self._by_end.get(addr, [])

    def branches_to(self, addr: int) -> List:
        return self._branches_to.get(addr, [])


def _single_ending_at(view: CodeView, addr: int):
    """The one instruction that ends at `addr`; None if none or ambiguous."""
    found = view.ending_at(addr)
    if len(found) != 1:
        return None
    return found[0]


def _fallthrough(view: CodeView, addr: int, max_pad: int = 8):
    """How control can arrive at `addr` from the instruction before it.

    Returns ("none", None) when the preceding real instruction is a jmp or
    ret, ("jcc", insn) when it is a conditional branch (its not-taken edge
    lands here), ("insn", insn) for an ordinary straight-line predecessor,
    and ("unknown", None) when the bytes before `addr` are not a single
    coherent decode. Alignment padding between them is stepped over.
    """
    cur = addr
    for _ in range(max_pad + 1):
        prev = _single_ending_at(view, cur)
        if prev is None:
            return ("unknown", None)
        if prev.is_ret or prev.is_jump:
            return ("none", None)
        if prev.is_cond_jump:
            return ("jcc", prev)
        if _is_padding(prev):
            cur = prev.address
            continue
        return ("insn", prev)
    return ("unknown", None)


def _edge_range(view: CodeView, jcc, reg: str,
                taken: bool) -> Optional[Tuple[int, int]]:
    """Index range on one edge out of a conditional branch."""
    if not jcc.is_cond_jump:
        return None
    # The flags must come from the instruction right before the branch and
    # from nowhere else: a jump landing on the jcc would bring its own.
    if view.branches_to(jcc.address):
        return None
    setter = _single_ending_at(view, jcc.address)
    if setter is None:
        return None
    ri = _reg_imm(setter)
    if ri is None or ri[0] != reg:
        return None
    n = ri[1]
    op = _base_mnemonic(setter)
    cc = jcc.mnemonic.lower()
    if op == "cmp":
        if (taken and cc in _JB) or (not taken and cc in _JAE):
            if 1 <= n <= MAX_ENTRIES:
                return (0, n - 1)
        elif (taken and cc in _JBE) or (not taken and cc in _JA):
            if 0 <= n < MAX_ENTRIES:
                return (0, n)
        return None
    if op == "sub":
        # Unsigned borrow means the value before the subtract was below n,
        # so the value after it is n below that: -n .. -1.
        if (taken and cc in _JB) or (not taken and cc in _JAE):
            if 1 <= n <= MAX_ENTRIES:
                return (-n, -1)
        return None
    return None


def _mask_bound(insn, reg: str) -> Optional[int]:
    """N when `insn` is `and reg, N` on the whole register, else None."""
    if _base_mnemonic(insn) != "and":
        return None
    ri = _reg_imm(insn)
    if ri is None or ri[0] != reg or not 0 <= ri[1] < MAX_ENTRIES:
        return None
    return ri[1]


def _negated(lo: int, hi: int, negations: int) -> Tuple[int, int]:
    for _ in range(negations):
        lo, hi = -hi, -lo
    return lo, hi


def index_range(view: CodeView, dispatch, reg: str,
                max_block: int = 16) -> Optional[Tuple[int, int]]:
    """Proven inclusive range of `reg` at `dispatch`, or None if unproven."""
    negations = 0
    cur = dispatch
    for _ in range(max_block):
        if view.branches_to(cur.address):
            kind, pred = _fallthrough(view, cur.address)
            break
        kind, pred = _fallthrough(view, cur.address)
        if kind != "insn":
            break
        # Straight-line predecessor inside the same block.
        if pred.is_call:
            return None
        if _writes(pred, reg):
            if (_base_mnemonic(pred) == "neg"
                    and pred.op_str.strip() == reg):
                negations += 1
            else:
                # `and reg, N` bounds the index by itself, whatever it was
                # on the way in; any other write leaves it unknown.
                mask = _mask_bound(pred, reg)
                if mask is None:
                    return None
                return _negated(0, mask, negations)
        cur = pred
    else:
        return None

    # `cur` starts the block. Every way in has to bound the index.
    ranges = []
    for branch in view.branches_to(cur.address):
        r = _edge_range(view, branch, reg, taken=True)
        if r is None:
            return None
        ranges.append(r)
    if kind == "jcc":
        r = _edge_range(view, pred, reg, taken=False)
        if r is None:
            return None
        ranges.append(r)
    elif kind != "none":
        return None
    if not ranges:
        return None

    return _negated(min(r[0] for r in ranges), max(r[1] for r in ranges),
                    negations)


def proven_index_span(view: CodeView, dispatch,
                      read_u32: Callable[[int], Optional[int]],
                      valid: Callable[[int], bool]
                      ) -> Optional[JumpTableSpan]:
    """The table a dispatch reads, where a forward read from disp is wrong.

    `valid` is the caller's test for a code pointer its forward reader would
    accept. Returns None for anything that is not `jmp [reg*4 + disp]`,
    whose index range cannot be proven, or whose table a forward read gets
    right: a non-negative range with a code pointer at disp is an ordinary
    switch, and the forward readers handle it.
    """
    parsed = parse_indexed_dispatch(dispatch)
    if parsed is None:
        return None
    reg, disp = parsed
    rng = index_range(view, dispatch, reg)
    if rng is None:
        return None
    lo, hi = rng
    if hi - lo + 1 > MAX_ENTRIES:
        return None
    if lo >= 0:
        first = read_u32(disp)
        if first is not None and valid(first):
            return None

    def holds_code(slot):
        return slot[1] is not None and valid(slot[1])

    slots = []
    for i in range(lo, hi + 1):
        va = (disp + 4 * i) & 0xFFFFFFFF
        slots.append((va, read_u32(va)))
    while slots and not holds_code(slots[0]):
        slots.pop(0)
        lo += 1
    while slots and not holds_code(slots[-1]):
        slots.pop()
        hi -= 1
    if not slots or not all(holds_code(s) for s in slots):
        return None
    span = JumpTableSpan(dispatch.address, disp, lo, hi, slots)
    # A table cannot overlap the instruction that reads it.
    if span.start < dispatch.end_address and dispatch.address < span.end:
        return None
    return span
