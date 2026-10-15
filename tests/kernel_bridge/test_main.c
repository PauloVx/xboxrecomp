/*
 * The kernel bridge.
 *
 * A title does not call the kernel by name. Its indirect calls land on a
 * thunk table that xbox_kernel_bridge_init() has rewritten to synthetic VAs,
 * and kernel_thunk_dispatch turns each of those into a bridge_* function.
 * That is the path the runtime actually executes, and it is easy to leave
 * untested: bridge_* functions are static and read guest CPU state rather
 * than taking arguments, so they cannot be called directly.
 *
 * They are reachable all the same. recomp_lookup_kernel is not static: given
 * a buffer standing in for guest memory it selects the slot and hands back
 * the dispatcher. tests/kernel_directory uses the same seam for the directory
 * ABI; this one covers kernel memory ordinals.
 *
 * Checked here with ordinal 173, MmGetPhysicalAddress, which had two
 * implementations that disagreed -- the bridge translating through the
 * contiguous window and xbox_MmGetPhysicalAddress returning its argument
 * unchanged. The bridge now calls the latter, and that delegation is what
 * these checks pin: asserting only on xbox_MmGetPhysicalAddress leaves a
 * bridge that stops delegating entirely green.
 */
/* kernel.h brings the NT type vocabulary: <windows.h> on Windows, the
 * shim in platform/xbox_winnt.h elsewhere. Nothing below is
 * platform-specific. */
#include "kernel.h"
#include "xbox_memory_layout.h"   /* RECOMP_TLS */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>

/* Provided by the generated title; a regression harness has no title, and a
 * static archive resolves only what gets pulled in. Nothing here calls a
 * guest function. */
typedef void (*recomp_func_t)(void);
recomp_func_t recomp_lookup(uint32_t xbox_va);
recomp_func_t recomp_lookup(uint32_t xbox_va) { (void)xbox_va; return NULL; }
recomp_func_t recomp_lookup_manual(uint32_t xbox_va);
recomp_func_t recomp_lookup_manual(uint32_t xbox_va) { (void)xbox_va; return NULL; }

/* The seam, plus the guest CPU state the dispatcher reads and writes. */
extern recomp_func_t recomp_lookup_kernel(uint32_t xbox_va);
extern RECOMP_TLS uint32_t g_eax, g_esp;
extern ptrdiff_t g_xbox_mem_offset;

static int failures;

static void check(int ok, const char *what, const char *detail)
{
    printf("  %-58s %s", what, ok ? "PASS" : "FAIL");
    if (!ok && detail) printf(" -- %s", detail);
    putchar('\n');
    if (!ok) failures++;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    check(xbox_kernel_thunk_entry_is_empty(0),
          "zero-filled kernel slot is empty", "zero should terminate the import run");
    check(xbox_kernel_thunk_entry_is_empty(0x80000000u),
          "marked ordinal-zero kernel slot is empty",
          "the marker bit alone must not make ordinal zero an import");
    check(!xbox_kernel_thunk_entry_is_empty(0x80000001u),
          "nonzero kernel ordinal remains an import",
          "a valid encoded ordinal was mistaken for an empty slot");

    const uint32_t THUNK_VA = 0x10000, STACK_VA = 0x20000;
    const uint32_t BASE_PTR_VA = 0x30000, SIZE_PTR_VA = 0x30004;

    uint8_t *mem = VirtualAlloc(NULL, 16 * 1024 * 1024,
                                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    check(mem != NULL, "guest memory stand-in allocated", NULL);
    if (!mem) {
        printf("\n%d failure(s); the rest needs guest memory.\n", failures);
        return 1;
    }
    g_xbox_mem_offset = (ptrdiff_t)mem;

    /* One import, in the form an XBE stores it: 0x80000000 | ordinal. */
    *(uint32_t *)(mem + THUNK_VA - 0x10) = 0x80000000u | 52u;
    *(uint32_t *)(mem + THUNK_VA - 0x20) = 0x80000000u; /* empty ordinal-zero slot */
    *(uint32_t *)(mem + THUNK_VA - 0x30) = 0x80000000u | 1357244544u;
    *(uint32_t *)(mem + THUNK_VA) = 0x80000000u | 173u;
    *(uint32_t *)(mem + THUNK_VA + 4) = 0x80000000u | 199u;
    *(uint32_t *)(mem + THUNK_VA + 8) = 0x80000000u | 2u;
    *(uint32_t *)(mem + THUNK_VA + 12) = 0x80000000u; /* unused declared slot */
    xbox_kernel_set_thunk_address(THUNK_VA, 3);
    xbox_kernel_bridge_init();

    /* The bounded lookbehind exists for real XBE runs (e.g. ordinals 52,
     * 51, and 5), but arbitrary data can also resemble a thunk marker. */
    uint32_t valid_preceding = *(uint32_t *)(mem + THUNK_VA - 0x10);
    uint32_t empty_preceding = *(uint32_t *)(mem + THUNK_VA - 0x20);
    uint32_t invalid_preceding = *(uint32_t *)(mem + THUNK_VA - 0x30);
    uint32_t empty_declared = *(uint32_t *)(mem + THUNK_VA + 12);
    check(valid_preceding == 0xFE000000u + 7 * 4,
          "valid preceding ordinal is bridged",
          "ordinal 52 was not assigned lookbehind slot 7");
    check(empty_preceding == 0x80000000u,
          "ordinal-zero lookbehind slot remains untouched",
          "an empty slot was treated as a kernel import");
    check(invalid_preceding == (0x80000000u | 1357244544u),
          "out-of-range preceding marker remains untouched",
          "a non-kernel value was treated as an import ordinal");
    check(empty_declared == 0x80000000u,
          "ordinal-zero declared slot remains untouched",
          "an empty declared slot was assigned a synthetic thunk");

    /* bridge_init rewrites the entry in place to a synthetic VA. */
    recomp_func_t fn = recomp_lookup_kernel(*(uint32_t *)(mem + THUNK_VA));
    check(fn != NULL, "ordinal 173 resolves to a bridge entry point",
          "the thunk entry was not rewritten to a synthetic VA");

    if (fn) {
        /* stdcall: the guest pushed a return address, then the argument. */
        uint32_t *sp = (uint32_t *)(mem + STACK_VA);
        sp[0] = 0xBEEF0001u;          /* guest return address */
        sp[1] = 0x80001000u;          /* inside the contiguous window */
        g_esp = STACK_VA;
        g_eax = 0xDEADBEEFu;
        fn();

        char detail[128];
        snprintf(detail, sizeof detail,
                 "bridge returned 0x%08X, expected 0x00001000", g_eax);
        check(g_eax == 0x00001000u,
              "ordinal 173 translates through the contiguous window", detail);

        /* And the argument must come off the guest stack. A wrong
         * stdcall_args_for_ordinal entry corrupts the caller's frame away
         * from the call, with nothing naming the ordinal. */
        snprintf(detail, sizeof detail, "esp is 0x%08X, expected 0x%08X",
                 g_esp, STACK_VA + 8);
        check(g_esp == STACK_VA + 8,
              "ordinal 173 pops its stdcall argument", detail);
    }

    /* Xbox uses 32-bit guest pointers and SIZE_T, even when the host is
     * 64-bit. Decommit is a successful no-op because mapped guest RAM is
     * already committed as one host region. */
    *(uint32_t *)(mem + BASE_PTR_VA) = 0x00100000u;
    *(uint32_t *)(mem + SIZE_PTR_VA) = 0x1000u;
    uint32_t *free_sp = (uint32_t *)(mem + STACK_VA);
    free_sp[0] = 0xBEEF0002u;
    free_sp[1] = BASE_PTR_VA;
    free_sp[2] = SIZE_PTR_VA;
    free_sp[3] = MEM_DECOMMIT;
    g_esp = STACK_VA;
    recomp_func_t free_fn = recomp_lookup_kernel(*(uint32_t *)(mem + THUNK_VA + 4));
    check(free_fn != NULL, "ordinal 199 resolves to a bridge entry point",
          "the free thunk entry was not rewritten to a synthetic VA");
    if (free_fn) {
        free_fn();
        check(g_eax == 0, "NtFreeVirtualMemory decommit succeeds for guest RAM",
              "the bridge passed a 32-bit guest pointer to a host pointer API");
        check(*(uint32_t *)(mem + BASE_PTR_VA) == 0x00100000u,
              "NtFreeVirtualMemory decommit preserves the guest base address",
              "decommit must not turn the base pointer into a host pointer");
        check(g_esp == STACK_VA + 16,
              "ordinal 199 pops its three stdcall arguments",
              "the guest stack was not advanced by 16 bytes");
    }

    recomp_func_t av_fn = recomp_lookup_kernel(*(uint32_t *)(mem + THUNK_VA + 8));
    check(av_fn != NULL, "ordinal 2 resolves to an AV bridge entry point", NULL);
    if (av_fn) {
        const uint32_t RESULT_VA = 0x40000;
        uint32_t *av_sp = (uint32_t *)(mem + STACK_VA);
        FILE *capture = fopen("av_option_test.log", "w");
        int saved_stderr = _dup(_fileno(stderr));
        check(capture != NULL && saved_stderr >= 0,
              "AV warning capture is available", NULL);
        if (capture && saved_stderr >= 0)
            _dup2(_fileno(capture), _fileno(stderr));
        av_sp[0] = 0xBEEF0003u;
        av_sp[1] = 0;          /* RegisterBase */
        av_sp[2] = 0x0Fu;      /* AV_OPTION_GUESS_FIELD */
        av_sp[3] = 0;          /* Param */
        av_sp[4] = RESULT_VA;
        *(uint32_t *)(mem + RESULT_VA) = 0xFFFFFFFFu;
        g_esp = STACK_VA;
        av_fn();
        fflush(stderr);
        if (saved_stderr >= 0) {
            _dup2(saved_stderr, _fileno(stderr));
            _close(saved_stderr);
        }
        if (capture) fclose(capture);
        check(*(uint32_t *)(mem + RESULT_VA) == 0,
              "progressive output reports no interlaced field", NULL);
        FILE *captured = fopen("av_option_test.log", "r");
        char av_log[1024] = {0};
        if (captured) {
            fread(av_log, 1, sizeof(av_log) - 1, captured);
            fclose(captured);
        }
        check(strstr(av_log, "AvSendTVEncoderOption: unknown option") == NULL,
              "AV option 0x0F is recognized without an unknown-option warning",
              "the guest uses this query while initializing D3D");
        check(g_esp == STACK_VA + 20,
              "ordinal 2 pops its four stdcall arguments", NULL);
    }

    xbox_kernel_init();
    check(xbox_kernel_thunk_table[147] == 0,
          "kernel init leaves unused fallback capacity empty",
          "the zero-filled fallback tail was treated as ordinal 0");
    xbox_kernel_shutdown();

    FILE *kernel_log = fopen("xbox_kernel.log", "r");
    unsigned ordinal_zero_errors = 0;
    if (kernel_log) {
        char line[512];
        while (fgets(line, sizeof(line), kernel_log))
            if (strstr(line, "Unresolved kernel ordinal 0"))
                ordinal_zero_errors++;
        fclose(kernel_log);
    }
    check(kernel_log != NULL && ordinal_zero_errors == 0,
          "kernel init emits no false ordinal-zero errors",
          "unused zero-filled fallback entries were resolved");

    VirtualFree(mem, 0, MEM_RELEASE);
    printf("\n%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
