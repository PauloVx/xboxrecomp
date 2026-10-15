/*
 * Thread placement on hybrid CPUs.
 *
 * The recompiled game runs a handful of threads that each need a whole fast core (the guest main thread, the NV2A
 * pusher and its D3D submitter) next to threads that poll or sleep (the register acknowledger, the vblank timer,
 * audio). On a CPU with performance and efficiency cores, and with SMT, the scheduler is free to put two of the busy
 * ones on the two hardware threads of one core, or one of them on an efficiency core: either costs a large part of
 * a frame's CPU time. xbox_PinThread(role) gives each role a core of its own.
 *
 *   RECOMP_THREAD_PLACEMENT=0       leave placement to the scheduler
 *   RECOMP_PIN_<ROLE>=<mask>        explicit affinity mask (hex, group 0) for a role, overriding the automatic choice
 *
 * Roles: MAIN, PUSHER, GPU, VBLANK, ACK, WORKER. Automatic placement needs a CPU with at least two efficiency
 * classes (a hybrid part): the busiest roles take performance cores in that order, one physical core each (both of its
 * hardware threads stay in the mask, so the thread may move between them but nothing else is put on that core on
 * purpose), and the polling roles take the efficiency cores.
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct PCore { KAFFINITY mask; BYTE eff; } PCore;

static PCore s_core[128];
static int s_ncores;
static BYTE s_max_eff, s_min_eff;
static int s_ready;
static int s_mode = -1;                 /* 0 off, 1 automatic */
static KAFFINITY s_pmask, s_emask;      /* every performance / efficiency hardware thread */
static int s_next_perf;

static void topology_init(void)
{
    DWORD len = 0;
    BYTE *buf, *p;
    s_ready = 1;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (!len) return;
    buf = (BYTE *)malloc(len);
    if (!buf) return;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        free(buf);
        return;
    }
    s_min_eff = 255;
    for (p = buf; p < buf + len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount >= 1 &&
            e->Processor.GroupMask[0].Group == 0 && s_ncores < 128) {
            s_core[s_ncores].mask = e->Processor.GroupMask[0].Mask;
            s_core[s_ncores].eff = e->Processor.EfficiencyClass;
            if (e->Processor.EfficiencyClass > s_max_eff) s_max_eff = e->Processor.EfficiencyClass;
            if (e->Processor.EfficiencyClass < s_min_eff) s_min_eff = e->Processor.EfficiencyClass;
            s_ncores++;
        }
        p += e->Size;
    }
    free(buf);
    if (s_max_eff > s_min_eff) {
        int i;
        for (i = 0; i < s_ncores; i++) {
            if (s_core[i].eff == s_max_eff) s_pmask |= s_core[i].mask;
            else s_emask |= s_core[i].mask;
        }
        /* The performance cores in logical-processor order. */
        for (i = 0; i < s_ncores; i++) {
            int j;
            for (j = i + 1; j < s_ncores; j++) {
                if (s_core[j].mask < s_core[i].mask) { PCore t = s_core[i]; s_core[i] = s_core[j]; s_core[j] = t; }
            }
        }
    }
}

static const char *const s_roles[] = { "MAIN", "PUSHER", "GPU", "VBLANK", "ACK", "WORKER" };

void xbox_PinThread(const char *role)
{
    char name[48];
    const char *e;
    KAFFINITY mask = 0;
    int r = -1, i;

    if (s_mode < 0) {
        const char *m = getenv("RECOMP_THREAD_PLACEMENT");
        s_mode = !(m && *m == '0');
    }
    for (i = 0; i < (int)(sizeof(s_roles) / sizeof(s_roles[0])); i++) if (!strcmp(role, s_roles[i])) r = i;
    snprintf(name, sizeof(name), "RECOMP_PIN_%s", role);
    e = getenv(name);
    if (e && *e) {
        mask = (KAFFINITY)strtoull(e, NULL, 16);
    } else if (s_mode) {
        static volatile LONG lock;
        while (InterlockedExchange(&lock, 1)) Sleep(0);
        if (!s_ready) topology_init();
        for (i = 0; i < (int)(sizeof(s_roles) / sizeof(s_roles[0])); i++) if (!strcmp(role, s_roles[i])) r = i;
        if (s_pmask && s_emask && r >= 0) {
            /* performance cores in order for the three busy roles; the rest share the efficiency cores */
            if (r <= 3) {
                int k = 0, want = r;      /* MAIN 0, PUSHER 1, GPU 2, VBLANK 3 */
                for (i = 0; i < s_ncores; i++) {
                    if (s_core[i].eff != s_max_eff) continue;
                    if (k++ == want) { mask = s_core[i].mask; break; }
                }
            } else {
                mask = s_emask;
            }
        }
        InterlockedExchange(&lock, 0);
    }
    if (mask) {
        KAFFINITY old = SetThreadAffinityMask(GetCurrentThread(), mask);
        static int said[8];
        if (r >= 0 && r < 8 && !said[r]) {
            said[r] = 1;
            fprintf(stderr, "[THREAD] %s on logical processors 0x%llX%s\n", role, (unsigned long long)mask, old ? "" : " (failed)");
        }
    }
}
