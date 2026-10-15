/* Per-draw GPU profile (diagnostic), included by kelvin_gpu.c after kelvin_gpu_time.inc.
 *
 * RECOMP_KGPU_GPUPROF=1 (with RECOMP_KGPU_GPUTIME set to any value, the frame brackets are shared): a timestamp
 * is written before every draw, clear and conversion pass; the time between two marks belongs to the pass that
 * started at the first. Passes are grouped by what they are (target size, combiner program, blend and depth
 * state, first texture size) and the periodic report lists the groups that cost the most GPU time per frame:
 *   [KGPUPROF] 1.92 ms/frame, 3.0 draws/frame, 4.1 Mtris: T 3440x1440 ps 3A9F0012 blend 1 z 1/0 tex0 640x480 ... */
#define KP_MAX_MARKS 6144
#define KP_CATS 512

typedef struct KpInfo {
    uint64_t key;
    uint32_t tw, th, tex0w, tex0h, icount, kind;
    uint32_t psh;
    uint8_t blend, ztest, zwrite, ntex;
} KpInfo;

typedef struct KpSlot {
    ID3D11Query *mark[KP_MAX_MARKS];
    ID3D11Query *endm[KP_MAX_MARKS];
    KpInfo info[KP_MAX_MARKS];
    unsigned int n;
} KpSlot;

typedef struct KpCat {
    KpInfo info;
    double ms;
    uint64_t draws, tris;
} KpCat;

static KpSlot *s_kp[KT_RING];
static KpCat s_kpc[KP_CATS];
static int s_kp_mode = -1;
static uint64_t s_kp_frames;

static bool kp_on(void)
{
    if (s_kp_mode < 0) {
        const char *e = getenv("RECOMP_KGPU_GPUPROF");
        s_kp_mode = e && *e && *e != '0';
    }
    return s_kp_mode == 1 && kt_on() && s_kt_open;
}

/* Called right before a pass is issued. kind: 0 draw, 1 clear, 2 conversion pass. */
static void kp_mark(const KpInfo *in)
{
    KtSlot *s;
    KpSlot *k;
    D3D11_QUERY_DESC qd;
    unsigned int slot;
    if (!kp_on()) return;
    slot = s_kt_cur % KT_RING;
    s = &s_kt[slot];
    if (!s_kp[slot]) s_kp[slot] = (KpSlot *)calloc(1, sizeof(KpSlot));
    k = s_kp[slot];
    if (!k || k->n >= KP_MAX_MARKS - 1) return;
    if (!k->mark[k->n]) {
        qd.MiscFlags = 0;
        qd.Query = D3D11_QUERY_TIMESTAMP;
        if (FAILED(ID3D11Device_CreateQuery(G.dev, &qd, &k->mark[k->n]))) return;
    }
    ID3D11DeviceContext_End(G.ctx, (ID3D11Asynchronous *)k->mark[k->n]);
    k->info[k->n] = *in;
    k->n++;
    (void)s;
}

/* Called right after the pass was issued: the time between the two marks is what the pass itself kept the GPU
 * busy, without the gaps where the GPU waited for the CPU to submit the next one. */
static void kp_done(void)
{
    KpSlot *k;
    D3D11_QUERY_DESC qd;
    unsigned int slot;
    if (!kp_on()) return;
    slot = s_kt_cur % KT_RING;
    k = s_kp[slot];
    if (!k || !k->n) return;
    if (!k->endm[k->n - 1]) {
        qd.MiscFlags = 0;
        qd.Query = D3D11_QUERY_TIMESTAMP;
        if (FAILED(ID3D11Device_CreateQuery(G.dev, &qd, &k->endm[k->n - 1]))) return;
    }
    ID3D11DeviceContext_End(G.ctx, (ID3D11Asynchronous *)k->endm[k->n - 1]);
    k->info[k->n - 1].ntex |= 0;   /* keeps the layout; the end mark exists for index n - 1 */
    k->info[k->n - 1].kind |= 0x100u;
}

static uint64_t kp_key(const KpInfo *i)
{
    uint64_t h = 1469598103934665603ULL;
    uint32_t v[9];
    unsigned int k;
    v[0] = i->tw; v[1] = i->th; v[2] = i->tex0w; v[3] = i->tex0h; v[4] = i->kind; v[5] = i->psh;
    v[6] = (uint32_t)i->blend | ((uint32_t)i->ztest << 8) | ((uint32_t)i->zwrite << 16) | ((uint32_t)i->ntex << 24);
    v[7] = v[8] = 0;
    for (k = 0; k < 7; k++) h = (h ^ v[k]) * 1099511628211ULL;
    return h;
}

/* A finished frame: read its marks, charge each interval to the pass that began it. */
static void kp_collect(unsigned int slot, UINT64 t_end, UINT64 freq)
{
    KpSlot *k = s_kp[slot];
    unsigned int i;
    if (!k || !k->n) return;
    for (i = 0; i < k->n; i++) {
        UINT64 t0 = 0, t1 = t_end;
        double ms;
        unsigned int c, probes;
        uint64_t key;
        if (ID3D11DeviceContext_GetData(G.ctx, (ID3D11Asynchronous *)k->mark[i], &t0, sizeof(t0), 0) != S_OK) continue;
        if (k->info[i].kind & 0x100u) {
            if (ID3D11DeviceContext_GetData(G.ctx, (ID3D11Asynchronous *)k->endm[i], &t1, sizeof(t1), 0) != S_OK) continue;
            k->info[i].kind &= ~0x100u;
        } else if (i + 1 < k->n && ID3D11DeviceContext_GetData(G.ctx, (ID3D11Asynchronous *)k->mark[i + 1], &t1, sizeof(t1), 0) != S_OK) continue;
        ms = (double)(t1 - t0) * 1000.0 / (double)freq;
        key = kp_key(&k->info[i]);
        for (probes = 0, c = (unsigned int)(key % KP_CATS); probes < KP_CATS; probes++, c = (c + 1) % KP_CATS) {
            if (!s_kpc[c].draws) { s_kpc[c].info = k->info[i]; s_kpc[c].info.key = key; break; }
            if (s_kpc[c].info.key == key) break;
        }
        if (probes == KP_CATS) continue;
        s_kpc[c].ms += ms;
        s_kpc[c].draws++;
        s_kpc[c].tris += k->info[i].icount / 3;
    }
    k->n = 0;
    s_kp_frames++;
}

static void kp_report(void)
{
    unsigned int shown, i, j;
    unsigned int order[KP_CATS], n = 0;
    double total = 0.0;
    if (!kp_on() && s_kp_mode != 1) return;
    if (!s_kp_frames) return;
    for (i = 0; i < KP_CATS; i++) if (s_kpc[i].draws) { order[n++] = i; total += s_kpc[i].ms; }
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (s_kpc[order[j]].ms > s_kpc[order[i]].ms) { unsigned int t = order[i]; order[i] = order[j]; order[j] = t; }
    fprintf(stderr, "[KGPUPROF] %.2f ms/frame over %llu frames; top groups:\n", total / (double)s_kp_frames,
            (unsigned long long)s_kp_frames);
    for (shown = 0; shown < n && shown < 14; shown++) {
        const KpCat *c = &s_kpc[order[shown]];
        static const char *const kinds[] = { "draw", "clear", "pass" };
        fprintf(stderr, "[KGPUPROF] %5.2f ms/frame %5.1f %s/frame %6.0f ktris/frame: T %ux%u ps %08X blend %u z %u/%u tex%u %ux%u\n",
                c->ms / (double)s_kp_frames, (double)c->draws / (double)s_kp_frames, kinds[c->info.kind % 3],
                (double)c->tris / (double)s_kp_frames / 1000.0, c->info.tw, c->info.th, c->info.psh, c->info.blend,
                c->info.ztest, c->info.zwrite, c->info.ntex, c->info.tex0w, c->info.tex0h);
    }
    memset(s_kpc, 0, sizeof(s_kpc));
    s_kp_frames = 0;
}
