/* Inclusive profile of one thread (diagnostic), included by kelvin_gpu.c after kelvin_gpu_slow.inc.
 *
 * RECOMP_KGPU_PROFI=start,seconds[,tid]: sample one thread (default: the guest main thread, whose id the game
 * records in g_black_main_tid from its Present callback) about every millisecond. Each sample's host stack is
 * scanned for return addresses inside the executable's code; every distinct function found counts once, so the
 * report lists, for the functions that account for the most time, the share of samples that were inside them or
 * inside something they called:
 *   [KPROFI] 61.2% sub_0020F6A0 ...   (exclusive: [KPROFI] self 9.1% sub_000205A0 ...)
 * Needs the PDB beside the executable. The generated code is one host function per guest function, so the
 * inclusive times are the game's own call tree. */
extern volatile DWORD g_black_main_tid;
extern volatile DWORD g_kelvin_pusher_tid;

typedef struct InclEntry { char name[64]; uint32_t incl, self; } InclEntry;

static DWORD WINAPI incl_thread(LPVOID arg)
{
    unsigned int start = ((unsigned int)(uintptr_t)arg) & 0xFFFF, secs = ((unsigned int)(uintptr_t)arg) >> 16;
    InclEntry *tab = (InclEntry *)calloc(4096, sizeof(InclEntry));
    struct { uint64_t addr; int16_t entry; } cache[8192];
    unsigned int ntab = 0, i, samples = 0;
    InclEntry *lines = (InclEntry *)calloc(8192, sizeof(InclEntry));
    unsigned int nlines = 0;
    DWORD tid = 0;
    int pusher = 0;
    HANDLE h;
    LARGE_INTEGER f;
    long long deadline;
    uint64_t mod_lo = (uint64_t)(uintptr_t)GetModuleHandleA(NULL), mod_hi, code_hi;
    const char *e = getenv("RECOMP_KGPU_PROFI");
    unsigned int top_n = getenv("RECOMP_KGPU_PROFILE_TOP") ? (unsigned int)atoi(getenv("RECOMP_KGPU_PROFILE_TOP")) : 60;
    if (top_n < 10) top_n = 10;
    if (top_n > 400) top_n = 400;
    {
        const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)(uintptr_t)mod_lo;
        const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const uint8_t *)dos + dos->e_lfanew);
        mod_hi = mod_lo + nt->OptionalHeader.SizeOfImage;
        code_hi = mod_lo + nt->OptionalHeader.BaseOfCode + nt->OptionalHeader.SizeOfCode;
    }
    memset(cache, 0, sizeof(cache));
    if (e) {
        const char *c = strchr(e, ',');
        if (c && (c = strchr(c + 1, ','))) {
            c++;
            if (!strncmp(c, "pusher", 6)) {
                pusher = 1;
            } else tid = (DWORD)atoi(c);
        }
    }
    QueryPerformanceFrequency(&f);
    Sleep(start * 1000);
    if (pusher) tid = g_kelvin_pusher_tid;
    if (!tid) tid = g_black_main_tid;
    if (!tid) { fprintf(stderr, "[KPROFI] no guest main thread id yet\n"); return 0; }
    h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, tid);
    if (!h) return 0;
    deadline = qpc_now() + (long long)secs * f.QuadPart;
    while (qpc_now() < deadline) {
        CONTEXT ctx;
        ctx.ContextFlags = CONTEXT_CONTROL;
        if (SuspendThread(h) != (DWORD)-1) {
            if (GetThreadContext(h, &ctx)) {
                uint64_t stack[768], seen_idx[64];
                SIZE_T got = 0;
                unsigned int k, nseen = 0;
                uint64_t addrs[64];
                unsigned int naddr = 0;
                addrs[naddr++] = ctx.Rip;
                {
                    /* read in 512-byte pieces: a read that runs past the committed stack fails as a whole */
                    SIZE_T off = 0, piece;
                    got = 0;
                    for (off = 0; off < sizeof(stack); off += 512) {
                        if (!ReadProcessMemory(GetCurrentProcess(), (LPCVOID)(ctx.Rsp + off), (uint8_t *)stack + off, 512, &piece)) break;
                        got += piece;
                    }
                    for (k = 0; k < got / 8 && naddr < 64; k++)
                        if (stack[k] >= mod_lo && stack[k] < code_hi) addrs[naddr++] = stack[k];
                }
                ResumeThread(h);
                for (k = 0; k < naddr; k++) {
                    unsigned int slot = (unsigned int)((addrs[k] >> 4) * 2654435761u) % 8192, probes;
                    int16_t entry = -1;
                    for (probes = 0; probes < 8192; probes++, slot = (slot + 1) % 8192) {
                        if (!cache[slot].addr) {
                            char nm[64];
                            unsigned int j;
                            slow_name(addrs[k], nm, sizeof(nm), 0);
                            for (j = 0; j < ntab; j++) if (!strcmp(tab[j].name, nm)) break;
                            if (j == ntab && ntab < 4096) { snprintf(tab[ntab].name, sizeof(tab[ntab].name), "%s", nm); ntab++; }
                            cache[slot].addr = addrs[k]; cache[slot].entry = (int16_t)(j < ntab ? j : -1);
                            entry = cache[slot].entry;
                            break;
                        }
                        if (cache[slot].addr == addrs[k]) { entry = cache[slot].entry; break; }
                    }
                    if (entry < 0) continue;
                    {
                        unsigned int q, dup = 0;
                        for (q = 0; q < nseen; q++) if (seen_idx[q] == (uint64_t)entry) { dup = 1; break; }
                        if (!dup && nseen < 64) { seen_idx[nseen++] = (uint64_t)entry; tab[entry].incl++; }
                    }
                    if (k == 0) tab[entry].self++;
                }
                {   /* the top frame by source line */
                    char ln[64];
                    unsigned int j;
                    slow_name(addrs[0], ln, sizeof(ln), 1);
                    if (!strncmp(ln, "_NLG", 4) || ln[0] == '?') {
                        HMODULE hm = NULL;
                        char mn[MAX_PATH];
                        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)(uintptr_t)addrs[0], &hm) &&
                            GetModuleFileNameA(hm, mn, sizeof(mn))) {
                            const char *base = strrchr(mn, '\\');
                            snprintf(ln, sizeof(ln), "%s [%s+0x%llX]", ln, base ? base + 1 : mn, (unsigned long long)(addrs[0] - (uint64_t)(uintptr_t)hm));
                        }
                    }
                    for (j = 0; j < nlines; j++) if (!strcmp(lines[j].name, ln)) break;
                    if (j == nlines && nlines < 8192) { snprintf(lines[nlines].name, sizeof(lines[nlines].name), "%s", ln); nlines++; }
                    if (j < nlines) lines[j].self++;
                }
                samples++;
            } else {
                ResumeThread(h);
            }
        }
        Sleep(1);
    }
    for (i = 0; i < ntab; i++) {
        unsigned int j;
        for (j = i + 1; j < ntab; j++)
            if (tab[j].incl > tab[i].incl) { InclEntry t = tab[i]; tab[i] = tab[j]; tab[j] = t; }
        if (i >= top_n) break;
    }
    fprintf(stderr, "[KPROFI] thread %lu, %u samples (inclusive share, self share)\n", (unsigned long)tid, samples);
    for (i = 0; i < ntab && i < top_n; i++)
        fprintf(stderr, "[KPROFI] %5.1f%% %5.1f%% %s\n", 100.0 * tab[i].incl / (double)(samples ? samples : 1),
                100.0 * tab[i].self / (double)(samples ? samples : 1), tab[i].name);
    fflush(stderr);
    for (i = 0; i < nlines && i < top_n; i++) {
        unsigned int j;
        for (j = i + 1; j < nlines; j++) if (lines[j].self > lines[i].self) { InclEntry t = lines[i]; lines[i] = lines[j]; lines[j] = t; }
        fprintf(stderr, "[KPROFL] %5.1f%% %s\n", 100.0 * lines[i].self / (double)(samples ? samples : 1), lines[i].name);
    }
    fflush(stderr);
    CloseHandle(h);
    free(tab);
    free(lines);
    return 0;
}

/* RECOMP_EXC_STAT=1: which exceptions the process takes, by thread and faulting address. The handler runs
 * first in the chain and always lets the real handlers see the exception. */
typedef struct ExcStat { DWORD tid; DWORD code; uint64_t rip, addr; uint32_t n; } ExcStat;
static ExcStat s_exc[256];
static volatile LONG s_nexc;
static LONG CALLBACK exc_stat_veh(PEXCEPTION_POINTERS ep)
{
    DWORD tid = GetCurrentThreadId(), code = ep->ExceptionRecord->ExceptionCode;
    uint64_t rip = ep->ContextRecord->Rip, addr = ep->ExceptionRecord->NumberParameters > 1 ? ep->ExceptionRecord->ExceptionInformation[1] : 0;
    LONG i, n = s_nexc;
    addr &= ~(uint64_t)0xFFF;
    for (i = 0; i < n; i++) {
        if (s_exc[i].tid == tid && s_exc[i].code == code && s_exc[i].rip == rip && s_exc[i].addr == addr) {
            InterlockedIncrement((volatile LONG *)&s_exc[i].n);
            return EXCEPTION_CONTINUE_SEARCH;
        }
    }
    if (n < 256) {
        i = InterlockedIncrement(&s_nexc) - 1;
        if (i < 256) { s_exc[i].tid = tid; s_exc[i].code = code; s_exc[i].rip = rip; s_exc[i].addr = addr; s_exc[i].n = 1; }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static DWORD WINAPI exc_stat_thread(LPVOID arg)
{
    (void)arg;
    for (;;) {
        LONG i, n;
        Sleep(10000);
        n = s_nexc < 256 ? s_nexc : 256;
        for (i = 0; i < n; i++) {
            LONG j;
            unsigned long long tot = 0;
            for (j = 0; j < i; j++) if (s_exc[j].tid == s_exc[i].tid) break;
            if (j < i) continue;
            for (j = 0; j < n; j++) if (s_exc[j].tid == s_exc[i].tid) tot += s_exc[j].n;
            fprintf(stderr, "[KEXC] total tid %lu%s: %llu exceptions\n", (unsigned long)s_exc[i].tid,
                    s_exc[i].tid == g_black_main_tid ? " (main)" : s_exc[i].tid == g_kelvin_pusher_tid ? " (pusher)" : "", tot);
        }
        for (i = 0; i < n; i++) {
            if (s_exc[i].n >= 20) {
                char nm[64];
                slow_name(s_exc[i].rip, nm, sizeof(nm), 1);
                fprintf(stderr, "[KEXC] tid %lu%s code 0x%08lX rip %s fault page 0x%llX x%u\n", (unsigned long)s_exc[i].tid,
                        s_exc[i].tid == g_black_main_tid ? " (main)" : s_exc[i].tid == g_kelvin_pusher_tid ? " (pusher)" : "",
                        (unsigned long)s_exc[i].code, nm, (unsigned long long)s_exc[i].addr, s_exc[i].n);
            }
        }
        fflush(stderr);
    }
    return 0;
}

static void incl_maybe_start(void)
{
    const char *e = getenv("RECOMP_KGPU_PROFI");
    unsigned int a = 0, b = 0;
    if (getenv("RECOMP_EXC_STAT")) {
        AddVectoredExceptionHandler(1, exc_stat_veh);
        CloseHandle(CreateThread(NULL, 0, exc_stat_thread, NULL, 0, NULL));
    }
    if (e && sscanf(e, "%u,%u", &a, &b) >= 2 && b)
        CloseHandle(CreateThread(NULL, 0, incl_thread, (LPVOID)(uintptr_t)(((b & 0xFFFF) << 16) | (a & 0xFFFF)), 0, NULL));
}
