/* GPU frame timing (diagnostic), included by kelvin_gpu.c.
 *
 * RECOMP_KGPU_GPUTIME=ms: bracket each frame's GPU work (from its first draw or clear to its flip) with
 * D3D11 timestamp queries, collect them a few frames later without stalling, add the average and the
 * worst frame to the periodic [KGPU] report and print every frame that took longer than `ms`:
 *   [KGPU] GPU frame 8731: 31.2 ms (1920 draws)
 * Tells a GPU-bound hitch (the CPU then waits inside Map/Unmap, which the "upload" figure shows) from a
 * CPU-bound one. Unset, every hook is one test of a flag. The debug overlay (pc_debug.h) turns the measurement on
 * while it is shown, for its GPU row. */
#define KT_RING 8
typedef struct KtSlot {
    ID3D11Query *disjoint, *begin, *end;
    bool pending;
    uint64_t frame;
    unsigned int draws;
} KtSlot;
static KtSlot s_kt[KT_RING];
static int s_kt_mode = -1;                 /* -1 unread, 0 off, 1 on, 2 failed */
static double s_kt_thresh_ms;
static bool s_kt_open;
static unsigned int s_kt_cur, s_kt_draws;
static double s_kt_sum_ms, s_kt_max_ms;
static uint64_t s_kt_frames;
static double s_kt_view_ms;               /* for the overlay: since kt_take_ms */
static unsigned int s_kt_view_n;

static bool kt_on(void)
{
    if (s_kt_mode < 0) {
        const char *e = getenv("RECOMP_KGPU_GPUTIME");
        s_kt_mode = e && *e && *e != '0';
        s_kt_thresh_ms = e && *e ? atof(e) : 0.0;
    }
    return (s_kt_mode == 1 || (s_kt_mode == 0 && pc_debug_visible())) && G.ok;
}

static void kt_begin(void)
{
    KtSlot *s;
    D3D11_QUERY_DESC qd;
    if (!kt_on()) return;
    if (s_kt_open) { s_kt_draws++; return; }
    s = &s_kt[s_kt_cur % KT_RING];
    if (s->pending) return;                 /* the ring is full: skip this frame's measurement */
    if (!s->disjoint) {
        qd.MiscFlags = 0;
        qd.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        if (FAILED(ID3D11Device_CreateQuery(G.dev, &qd, &s->disjoint))) { s_kt_mode = 2; return; }
        qd.Query = D3D11_QUERY_TIMESTAMP;
        if (FAILED(ID3D11Device_CreateQuery(G.dev, &qd, &s->begin)) || FAILED(ID3D11Device_CreateQuery(G.dev, &qd, &s->end))) { s_kt_mode = 2; return; }
    }
    ID3D11DeviceContext_Begin(G.ctx, (ID3D11Asynchronous *)s->disjoint);
    ID3D11DeviceContext_End(G.ctx, (ID3D11Asynchronous *)s->begin);
    s_kt_open = true;
    s_kt_draws = 1;
}

static void kt_end(uint64_t frame)
{
    KtSlot *s;
    if (!s_kt_open) return;
    s = &s_kt[s_kt_cur % KT_RING];
    ID3D11DeviceContext_End(G.ctx, (ID3D11Asynchronous *)s->end);
    ID3D11DeviceContext_End(G.ctx, (ID3D11Asynchronous *)s->disjoint);
    s->pending = true;
    s->frame = frame;
    s->draws = s_kt_draws;
    s_kt_cur++;
    s_kt_open = false;
}

static void kp_collect(unsigned int slot, UINT64 t_end, UINT64 freq);
static void kp_report(void);

/* Collect finished frames, oldest first, without waiting. */
static void kt_poll(void)
{
    unsigned int i;
    if (!kt_on()) return;
    for (i = 0; i < KT_RING; i++) {
        KtSlot *s = &s_kt[(s_kt_cur + i) % KT_RING];      /* oldest first */
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
        UINT64 t0, t1;
        if (!s->pending) continue;
        if (ID3D11DeviceContext_GetData(G.ctx, (ID3D11Asynchronous *)s->disjoint, &dj, sizeof(dj), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) continue;
        if (ID3D11DeviceContext_GetData(G.ctx, (ID3D11Asynchronous *)s->begin, &t0, sizeof(t0), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) continue;
        if (ID3D11DeviceContext_GetData(G.ctx, (ID3D11Asynchronous *)s->end, &t1, sizeof(t1), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) continue;
        s->pending = false;
        if (!dj.Disjoint && dj.Frequency) kp_collect((s_kt_cur + i) % KT_RING, t1, dj.Frequency);
        if (!dj.Disjoint && dj.Frequency) {
            double ms = (double)(t1 - t0) * 1000.0 / (double)dj.Frequency;
            s_kt_sum_ms += ms;
            s_kt_view_ms += ms;
            s_kt_view_n++;
            if (ms > s_kt_max_ms) s_kt_max_ms = ms;
            s_kt_frames++;
            if (s_kt_thresh_ms > 1.0 && ms > s_kt_thresh_ms)
                fprintf(stderr, "[KGPU] GPU frame %llu: %.1f ms (%u draws)\n", (unsigned long long)s->frame, ms, s->draws);
        }
    }
}

/* The average GPU time of the frames measured since the last call, -1 when none were. */
static double kt_take_ms(void)
{
    const double ms = s_kt_view_n ? s_kt_view_ms / s_kt_view_n : -1.0;
    s_kt_view_ms = 0.0;
    s_kt_view_n = 0;
    return ms;
}

static void kt_report(void)
{
    if (s_kt_mode != 1 || !G.ok || !s_kt_frames) return;
    fprintf(stderr, "[KGPU] GPU time per frame: avg %.2f ms, worst %.2f ms over %llu frames\n",
            s_kt_sum_ms / (double)s_kt_frames, s_kt_max_ms, (unsigned long long)s_kt_frames);
    s_kt_sum_ms = s_kt_max_ms = 0.0;
    s_kt_frames = 0;
    kp_report();
}
