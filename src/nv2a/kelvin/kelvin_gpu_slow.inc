/* Slow-flip sampler (diagnostic), included by kelvin_gpu.c after the sampling profiler.
 *
 * RECOMP_KGPU_SLOW=ms[,start_seconds]: sample the instruction pointer of the busiest threads (cumulative
 * CPU time, at most six) every millisecond, and for each flip interval longer than `ms` print what each
 * of them was doing in it: "[KSLOW] flip N 87.3 ms, tid T (80 samples): sub_0012ABCD 60% ...". Finds the
 * cause of single slow frames, which an aggregate profile averages away. Needs the PDB beside the .exe. */
#define SLOW_UP 6
typedef struct SlowSample { long long t; uint64_t rip, up[SLOW_UP]; unsigned int th; } SlowSample;
#define SLOW_RING 65536
#define SLOW_MAX_THREADS 8

static void slow_name(uint64_t addr, char *out, size_t cap, int with_line)
{
    char buf[sizeof(SYMBOL_INFO) + 128];
    SYMBOL_INFO *si = (SYMBOL_INFO *)buf;
    DWORD64 disp = 0;
    memset(buf, 0, sizeof(buf));
    si->SizeOfStruct = sizeof(SYMBOL_INFO);
    si->MaxNameLen = 120;
    if (SymFromAddr(GetCurrentProcess(), addr, &disp, si)) {
        IMAGEHLP_LINE64 line;
        DWORD ldisp = 0;
        memset(&line, 0, sizeof(line));
        line.SizeOfStruct = sizeof(line);
        if (with_line && SymGetLineFromAddr64(GetCurrentProcess(), addr, &ldisp, &line)) snprintf(out, cap, "%s@%lu", si->Name, (unsigned long)line.LineNumber);
        else snprintf(out, cap, "%s", si->Name);
    } else snprintf(out, cap, "?%llX", (unsigned long long)(addr >> 12));
}

static void slow_report(const SlowSample *ring, uint64_t wrote, const DWORD *tids, long long t0, long long t1, uint64_t flip, double ms)
{
    unsigned int th;
    uint64_t first = wrote > SLOW_RING ? wrote - SLOW_RING : 0, i;
    for (th = 0; th < SLOW_MAX_THREADS; th++) {
        typedef struct { char name[96]; unsigned int n; } Agg;
        Agg agg[192];
        unsigned int na = 0, total = 0, k;
        for (i = first; i < wrote; i++) {
            const SlowSample *x = &ring[i % SLOW_RING];
            char nm[128], a[48], u[SLOW_UP][40];
            unsigned int j, nu = 0, chosen = 0;
            size_t len;
            if (x->th != th || x->t <= t0 || x->t > t1) continue;
            total++;
            slow_name(x->rip, a, sizeof(a), 1);
            for (j = 0; j < SLOW_UP && x->up[j]; j++) slow_name(x->up[j], u[nu++], sizeof(u[0]), 0);
            len = (size_t)snprintf(nm, sizeof(nm), "%s", a);
            /* the first two guest functions (sub_...) on the stack, else the first two named frames */
            for (j = 0; j < nu && chosen < 2; j++)
                if (!strncmp(u[j], "sub_", 4)) { len += (size_t)snprintf(nm + len, sizeof(nm) - len, "<%s", u[j]); chosen++; }
            for (j = 0; j < nu && chosen < 2 && !chosen; j++)
                if (u[j][0] != '?') { len += (size_t)snprintf(nm + len, sizeof(nm) - len, "<%s", u[j]); chosen++; }
            (void)len;
            for (j = 0; j < na; j++) if (!strcmp(agg[j].name, nm)) break;
            if (j == na && na < 192) { snprintf(agg[na].name, sizeof(agg[na].name), "%s", nm); agg[na].n = 0; na++; }
            if (j < na) agg[j].n++;
        }
        if (!total) continue;
        for (k = 0; k < na; k++) {
            unsigned int j;
            for (j = k + 1; j < na; j++) if (agg[j].n > agg[k].n) { Agg tmp = agg[k]; agg[k] = agg[j]; agg[j] = tmp; }
        }
        fprintf(stderr, "[KSLOW] flip %llu %.1f ms, tid %lu (%u samples):", (unsigned long long)flip, ms, (unsigned long)tids[th], total);
        for (k = 0; k < na && k < 10; k++) fprintf(stderr, " %s %.0f%%", agg[k].name, 100.0 * agg[k].n / (double)total);
        fprintf(stderr, "\n");
    }
}

/* The busiest threads of the process by cumulative CPU time. */
static unsigned int slow_pick(DWORD self, DWORD pid, DWORD *tids, HANDLE *hs)
{
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te;
    struct { DWORD tid; ULONGLONG cpu; } best[SLOW_MAX_THREADS];
    static struct { DWORD tid; ULONGLONG cpu; } seen[1024];      /* CPU time at the previous pick, per thread */
    static unsigned int nseen;
    unsigned int nb = 0, nth = 0, i;
    te.dwSize = sizeof(te);
    if (snap != INVALID_HANDLE_VALUE && Thread32First(snap, &te)) {
        do {
            HANDLE h;
            FILETIME c, e, k, u;
            if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
            h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
            if (!h) continue;
            if (GetThreadTimes(h, &c, &e, &k, &u)) {
                ULONGLONG total = (((ULONGLONG)k.dwHighDateTime << 32) | k.dwLowDateTime) + (((ULONGLONG)u.dwHighDateTime << 32) | u.dwLowDateTime);
                ULONGLONG cpu = total;                    /* ranked by what it used since the last pick */
                unsigned int j, q;
                for (q = 0; q < nseen && seen[q].tid != te.th32ThreadID; q++) ;
                if (q < nseen) { cpu = total >= seen[q].cpu ? total - seen[q].cpu : 0; seen[q].cpu = total; }
                else if (nseen < 1024) { seen[nseen].tid = te.th32ThreadID; seen[nseen].cpu = total; nseen++; cpu = 0; }
                j = nb < SLOW_MAX_THREADS ? nb : SLOW_MAX_THREADS - 1;
                if (nb < SLOW_MAX_THREADS || cpu > best[j].cpu) {
                    best[j].tid = te.th32ThreadID; best[j].cpu = cpu;
                    if (nb < SLOW_MAX_THREADS) nb++;
                    while (j > 0 && best[j].cpu > best[j - 1].cpu) {
                        DWORD tt = best[j].tid; ULONGLONG cc = best[j].cpu;
                        best[j].tid = best[j - 1].tid; best[j].cpu = best[j - 1].cpu;
                        best[j - 1].tid = tt; best[j - 1].cpu = cc;
                        j--;
                    }
                }
            }
            CloseHandle(h);
        } while (Thread32Next(snap, &te));
    }
    if (snap != INVALID_HANDLE_VALUE) CloseHandle(snap);
    for (i = 0; i < SLOW_MAX_THREADS; i++) if (hs[i]) { CloseHandle(hs[i]); hs[i] = NULL; }
    for (i = 0; i < nb; i++) {
        HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, best[i].tid);
        if (h) { hs[nth] = h; tids[nth++] = best[i].tid; }
    }
    return nth;
}

static DWORD WINAPI slow_thread(LPVOID arg)
{
    unsigned int thr = ((unsigned int)(uintptr_t)arg) & 0xFFFF, start = ((unsigned int)(uintptr_t)arg) >> 16;
    SlowSample *ring = (SlowSample *)calloc(SLOW_RING, sizeof(SlowSample));
    DWORD tids[SLOW_MAX_THREADS] = { 0 };
    HANDLE hs[SLOW_MAX_THREADS] = { 0 };
    unsigned int nth = 0;
    uint64_t wrote = 0;
    long long last_pick = 0;
    LONG64 seen;
    DWORD pid = GetCurrentProcessId(), self = GetCurrentThreadId();
    uint64_t mod_lo = (uint64_t)(uintptr_t)GetModuleHandleA(NULL), mod_hi;
    if (!ring) return 0;
    {
        const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)(uintptr_t)mod_lo;
        const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const uint8_t *)dos + dos->e_lfanew);
        mod_hi = mod_lo + nt->OptionalHeader.SizeOfImage;
    }
    Sleep(start * 1000);
    seen = s_flip_count;
    for (;;) {
        long long now = qpc_now();
        unsigned int i;
        LONG64 fc;
        if (now - last_pick > (long long)G.qpf / 4) {
            last_pick = now;
            nth = slow_pick(self, pid, tids, hs);
        }
        for (i = 0; i < nth; i++) {
            CONTEXT ctx;
            ctx.ContextFlags = CONTEXT_CONTROL;
            if (SuspendThread(hs[i]) == (DWORD)-1) continue;
            if (GetThreadContext(hs[i], &ctx)) {
                SlowSample *x = &ring[wrote % SLOW_RING];
                uint64_t stack[512];
                SIZE_T got = 0;
                unsigned int k, found = 0;
                x->t = qpc_now(); x->rip = ctx.Rip; x->th = i;
                memset(x->up, 0, sizeof(x->up));
                /* the first few addresses inside the executable found on the stack (return addresses, mostly) */
                if (ReadProcessMemory(GetCurrentProcess(), (LPCVOID)ctx.Rsp, stack, sizeof(stack), &got)) {
                    for (k = 0; k < got / 8 && found < SLOW_UP; k++)
                        if (stack[k] >= mod_lo && stack[k] < mod_hi) x->up[found++] = stack[k];
                }
                wrote++;
            }
            ResumeThread(hs[i]);
        }
        fc = s_flip_count;
        while (seen < fc) {
            long long t0, t1;
            seen++;
            if (fc - seen >= 1000) continue;
            t0 = s_flip_q[(seen - 1) & 1023]; t1 = s_flip_q[seen & 1023];
            if (t0 && t1 > t0) {
                double ms = (double)(t1 - t0) * 1000.0 / (double)G.qpf;
                if (ms > (double)thr) slow_report(ring, wrote, tids, t0, t1, (uint64_t)seen, ms);
            }
        }
        Sleep(1);
    }
    return 0;
}

void kgpu_symbol_name(uint64_t addr, char *out, size_t cap) { slow_name(addr, out, cap, 0); }
