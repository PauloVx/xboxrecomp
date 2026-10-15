#!/usr/bin/env python3
import argparse, json, bisect, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

UNRESOLVED_PATH = os.path.join(SCRIPT_DIR, 'output', 'unresolved_symbols.txt')
FUNCTIONS_PATH = os.path.join(REPO_ROOT, 'tools', 'disasm', 'output', 'functions.json')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'output', 'missing_functions.json')

def section_start(section):
    value = section.get('virtual_addr', section.get('va'))
    return int(value, 16) if isinstance(value, str) else value


def section_size(section):
    return section.get('virtual_size', section.get('size'))


def find_section(addr, sections):
    for s in sections:
        start = section_start(s)
        if start <= addr < start + section_size(s):
            return s['name']
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Classify unresolved addresses using parsed XBE sections.')
    parser.add_argument('--xbe-analysis', required=True,
                        help='JSON metadata produced by tools.xbe_parser for the XBE used in this build')
    parser.add_argument('--unresolved', default=UNRESOLVED_PATH,
                        help='text file containing unresolved sub_XXXXXXXX addresses')
    parser.add_argument('--functions', default=FUNCTIONS_PATH,
                        help='function database JSON')
    parser.add_argument('--output-dir', default=os.path.dirname(OUTPUT_PATH),
                        help='directory for missing_functions.json and addable_functions.json')
    args = parser.parse_args(argv)
    with open(args.xbe_analysis, encoding='utf-8') as f:
        sections = json.load(f)['sections']
    text_sections = [s for s in sections if s['name'] == '.text']
    if len(text_sections) != 1:
        parser.error('XBE metadata must contain exactly one .text section')
    text_start = section_start(text_sections[0])
    text_end = text_start + section_size(text_sections[0])

    print('Loading function database...')
    with open(args.functions, encoding='utf-8') as f:
        raw_funcs = json.load(f)

    functions = []
    for func in raw_funcs:
        start = int(func['start'], 16)
        end = int(func['end'], 16)
        name = func['name']
        functions.append((start, end, name))

    functions.sort(key=lambda x: x[0])
    func_starts = [f[0] for f in functions]
    print(f'  Loaded {len(functions)} functions')
    print(f'  Range: 0x{functions[0][0]:08X} - 0x{functions[-1][1]:08X}')

    print('Loading unresolved symbols...')
    with open(args.unresolved, encoding='utf-8') as f:
        unresolved = []
        for line in f:
            line = line.strip()
            if not line: continue
            addr = int(line.replace('sub_', ''), 16)
            unresolved.append(addr)
    unresolved.sort()
    print(f'  Loaded {len(unresolved)} unresolved symbols')
    print(f'  Range: 0x{unresolved[0]:08X} - 0x{unresolved[-1]:08X}')

    results = {
        'mid_function': [],
        'continuation': [],
        'gap': [],
        'library_section': [],
        'data_section': [],
        'unknown': [],
    }
    missing_functions = []

    for addr in unresolved:
        section = find_section(addr, sections)

        if section is None:
            results['unknown'].append(addr)
            missing_functions.append({'address': f'0x{addr:08X}', 'type': 'unknown', 'estimated_end': None})
            continue

        if section in ('.rdata', '.data', '.data1'):
            results['data_section'].append(addr)
            missing_functions.append({'address': f'0x{addr:08X}', 'type': 'data_section', 'section': section, 'estimated_end': None})
            continue

        if section != '.text':
            sec_info = next(s for s in sections if s['name'] == section)
            sec_end = section_start(sec_info) + section_size(sec_info)
            results['library_section'].append((addr, section))
            missing_functions.append({'address': f'0x{addr:08X}', 'type': 'library_section', 'section': section, 'estimated_end': f'0x{sec_end:08X}'})
            continue

        idx = bisect.bisect_right(func_starts, addr) - 1

        if idx < 0:
            next_start = func_starts[0] if func_starts else text_end
            results['gap'].append((addr, text_start, next_start, next_start - addr))
            missing_functions.append({'address': f'0x{addr:08X}', 'type': 'gap', 'next_func_start': f'0x{next_start:08X}', 'estimated_end': f'0x{next_start:08X}', 'gap_size': next_start - addr})
            continue

        func_start, func_end, func_name = functions[idx]

        if func_start < addr < func_end:
            results['mid_function'].append((addr, func_start, func_name))
            missing_functions.append({'address': f'0x{addr:08X}', 'type': 'mid_function', 'parent_func': f'0x{func_start:08X}', 'parent_name': func_name, 'offset_into_func': addr - func_start, 'estimated_end': f'0x{func_end:08X}'})
            continue

        if addr == func_end:
            next_idx = idx + 1
            next_start = functions[next_idx][0] if next_idx < len(functions) else text_end
            gap_to_next = next_start - addr
            results['continuation'].append((addr, func_start, func_name, gap_to_next))
            missing_functions.append({'address': f'0x{addr:08X}', 'type': 'continuation', 'parent_func': f'0x{func_start:08X}', 'parent_name': func_name, 'estimated_end': f'0x{next_start:08X}', 'gap_size': gap_to_next})
            continue

        if addr == func_start:
            print(f'  WARNING: Unresolved 0x{addr:08X} matches known function {func_name}')
            continue

        next_idx = idx + 1
        next_start = functions[next_idx][0] if next_idx < len(functions) else text_end

        if func_end <= addr < next_start:
            gap_size = next_start - addr
            results['gap'].append((addr, func_end, next_start, gap_size))
            missing_functions.append({'address': f'0x{addr:08X}', 'type': 'gap', 'prev_func_end': f'0x{func_end:08X}', 'next_func_start': f'0x{next_start:08X}', 'estimated_end': f'0x{next_start:08X}', 'gap_size': gap_size})
            continue

        print(f'  WARNING: Could not classify 0x{addr:08X}')
        results['unknown'].append(addr)

    # Summary
    print()
    print('================================================================================')
    print('UNRESOLVED SYMBOL ANALYSIS SUMMARY')
    print('================================================================================')
    total = len(unresolved)
    print(f'\nTotal unresolved symbols: {total}')
    print()

    mid = results['mid_function']
    print(f'(a) MID-FUNCTION ENTRIES: {len(mid)} ({100*len(mid)/total:.1f}%)')
    print('    These addresses fall inside a known function body.')
    print('    Likely: tail-call targets, computed jumps, or function pointer offsets.')
    if mid:
        print('    Examples:')
        for addr, parent, name in mid[:10]:
            offset = addr - parent
            print(f'      0x{addr:08X}  (inside {name}, offset +0x{offset:X})')
    print()

    cont = results['continuation']
    print(f'(b) CONTINUATION PAST BOUNDARY: {len(cont)} ({100*len(cont)/total:.1f}%)')
    print('    Address falls exactly at a known function end.')
    print('    The function database underestimated the function size.')
    if cont:
        print('    Examples:')
        for addr, parent, name, gap in cont[:10]:
            print(f'      0x{addr:08X}  (after {name} ends, gap to next: {gap} bytes)')
        gaps = [g for _, _, _, g in cont]
        print(f'    Gap size stats: min={min(gaps)}, max={max(gaps)}, median={sorted(gaps)[len(gaps)//2]}, mean={sum(gaps)/len(gaps):.0f}')
    print()

    gap = results['gap']
    print(f'(c) GAP FUNCTIONS: {len(gap)} ({100*len(gap)/total:.1f}%)')
    print('    Addresses fall in gaps between known functions in .text.')
    print('    These are likely real functions the disassembler missed.')
    if gap and isinstance(gap[0], tuple):
        print('    Examples:')
        for item in gap[:10]:
            if len(item) == 4:
                a, pe, ns, gs = item
                print(f'      0x{a:08X}  (gap: 0x{pe:08X}-0x{ns:08X}, {gs} bytes)')
        gap_sizes = [item[3] for item in gap if isinstance(item, tuple) and len(item) == 4]
        if gap_sizes:
            print(f'    Gap size stats: min={min(gap_sizes)}, max={max(gap_sizes)}, median={sorted(gap_sizes)[len(gap_sizes)//2]}, mean={sum(gap_sizes)/len(gap_sizes):.0f}')
            print('    Gap size distribution:')
            for label, lo, hi in [('1-16 bytes',1,16),('17-64 bytes',17,64),('65-256 bytes',65,256),('257-1024 bytes',257,1024),('1025+ bytes',1025,999999999)]:
                count = sum(1 for s in gap_sizes if lo <= s <= hi)
                print(f'      {label:20s}: {count:5d}')
    print()

    lib = results['library_section']
    print(f'(d) LIBRARY SECTION FUNCTIONS: {len(lib)} ({100*len(lib)/total:.1f}%)')
    print('    Addresses fall in non-.text code sections (XDK libraries).')
    if lib:
        section_counts = {}
        for _, sec in lib:
            section_counts[sec] = section_counts.get(sec, 0) + 1
        print('    By section:')
        for sec, count in sorted(section_counts.items(), key=lambda x: -x[1]):
            sec_info = next(s for s in sections if s['name'] == sec)
            va = section_start(sec_info)
            sz = section_size(sec_info)
            print(f'      {sec:12s}: {count:4d} functions (section: 0x{va:08X}, {sz:,} bytes)')
        print('    Examples:')
        for a, sec in lib[:10]:
            print(f'      0x{a:08X}  ({sec})')
    print()

    data = results['data_section']
    print(f'(e) DATA SECTION REFERENCES: {len(data)} ({100*len(data)/total:.1f}%)')
    print('    Addresses fall in .rdata/.data - not code, likely misidentified.')
    if data:
        print('    Examples:')
        for a in data[:10]:
            sec = find_section(a, sections)
            print(f'      0x{a:08X}  ({sec})')
    print()

    unk = results['unknown']
    if unk:
        print(f'(f) UNKNOWN (outside all sections): {len(unk)}')
        for a in unk[:5]:
            print(f'      0x{a:08X}')
    print()

    # Actionable summary
    print('================================================================================')
    print('ACTIONABLE SUMMARY')
    print('================================================================================')
    text_actionable = len(gap) + len(cont)
    print(f'\n  Symbols in .text that can extend function DB: {text_actionable} (gap: {len(gap)}, continuation: {len(cont)})')
    print(f'  Symbols in library sections (need lib disassembly):  {len(lib)}')
    print(f'  Mid-function entries (need parent func extension):   {len(mid)}')
    print(f'  Data section refs (need investigation):              {len(data)}')
    print(f'  Unknown:                                             {len(unk)}')
    print()

    if gap and isinstance(gap[0], tuple):
        unique_gaps = set()
        for item in gap:
            if len(item) == 4:
                _, pe, ns, _ = item
                unique_gaps.add((pe, ns))
        print(f'  Unique inter-function gaps with unresolved symbols: {len(unique_gaps)}')
        total_gap_bytes = sum(ns - pe for pe, ns in unique_gaps)
        print(f'  Total bytes in those gaps: {total_gap_bytes:,} ({total_gap_bytes/1024:.1f} KB)')

    if cont:
        unique_parents = set(parent for _, parent, _, _ in cont)
        print(f'  Functions needing boundary extension: {len(unique_parents)}')
        total_cont_bytes = sum(g for _, _, _, g in cont)
        print(f'  Total continuation bytes: {total_cont_bytes:,} ({total_cont_bytes/1024:.1f} KB)')

    if lib:
        lib_sections_used = set(sec for _, sec in lib)
        total_lib_bytes = sum(section_size(next(s for s in sections if s['name'] == sec)) for sec in lib_sections_used)
        print(f'  Total library section bytes (all used sections): {total_lib_bytes:,} ({total_lib_bytes/1024:.1f} KB)')
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'missing_functions.json')
    print(f'\nWriting {len(missing_functions)} entries to {output_path}...')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(missing_functions, f, indent=2)
    print('Done.')

    quick_add_path = os.path.join(args.output_dir, 'addable_functions.json')
    addable = []
    for entry in missing_functions:
        if entry['type'] in ('gap', 'continuation', 'library_section'):
            addable.append({
                'address': entry['address'],
                'type': entry['type'],
                'estimated_end': entry.get('estimated_end'),
                'gap_size': entry.get('gap_size'),
                'section': entry.get('section', '.text'),
            })
    print(f'Writing {len(addable)} addable function candidates to {quick_add_path}...')
    with open(quick_add_path, 'w', encoding='utf-8') as f:
        json.dump(addable, f, indent=2)
    print('Done.')


if __name__ == '__main__':
    main()
