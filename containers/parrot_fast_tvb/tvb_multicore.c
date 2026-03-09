// THE VIRTUAL BRAIN -- C -- with electical activity and CBV and stimulus
// A fast implementation of TVB-style brain network models based on
// DYNAMIC MEAN FIELD MODEL Deco et al. 2014 Journal of Neuroscience
//
//  m.schirner@fu-berlin.de
//  michael.schirner@charite.de
//
// MIT LICENSE
// Copyright 2020 Michael Schirner
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

pthread_mutex_t mutex_thrcount;
pthread_barrier_t mybarrier_base, mybarrier1, mybarrier2, mybarrier3;

struct Xi_p { float **Xi_elems; };
struct SC_capS { float *cap; };
struct SC_inpregS { int *inpreg; };

// stimulus description
typedef struct {
    int   node;       // global node index [0..nodes-1]
    int   start_ts;   // start time, in simulation time steps (ms)
    int   end_ts;     // end time (exclusive), in simulation time steps (ms)
    float amplitude;  // stimulus amplitude
} Stim_t;

static inline int divRoundUp(const int x, const int y){ return (x + y - 1) / y; }

// load optional stimulus file
int load_stimuli(const char *fname, int nodes, int time_steps,
                 Stim_t **stimuli_out, int *n_stimuli_out)
{
    FILE *f = fopen(fname, "r");
    if (!f) {
        // no stim file -> no stimuli
        *stimuli_out = NULL;
        *n_stimuli_out = 0;
        return 0;
    }

    int n;
    if (fscanf(f, "%d", &n) != 1 || n <= 0) {
        fclose(f);
        *stimuli_out = NULL;
        *n_stimuli_out = 0;
        return 0;
    }

    Stim_t *stim = (Stim_t *)malloc((size_t)n * sizeof(Stim_t));
    if (!stim) {
        printf("ERROR: Out of memory while loading stimuli.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        int node;
        float start_ms, end_ms, amp;
        if (fscanf(f, "%d %f %f %f", &node, &start_ms, &end_ms, &amp) != 4) {
            printf("ERROR: Bad line in stimulus file %s (line %d)\n", fname, i + 2);
            fclose(f);
            free(stim);
            exit(EXIT_FAILURE);
        }
        if (node < 0 || node >= nodes) {
            printf("ERROR: Stimulus node %d out of range [0,%d)\n", node, nodes);
            fclose(f);
            free(stim);
            exit(EXIT_FAILURE);
        }

        // ts increments in ms, BOLD_TR is in ms, so treat times as ms
        int start_ts = (int)(start_ms + 0.5f);
        int end_ts   = (int)(end_ms   + 0.5f);
        if (start_ts < 0) start_ts = 0;
        if (end_ts > time_steps) end_ts = time_steps;

        stim[i].node      = node;
        stim[i].start_ts  = start_ts;
        stim[i].end_ts    = end_ts;
        stim[i].amplitude = amp;
    }

    fclose(f);
    *stimuli_out  = stim;
    *n_stimuli_out = n;
    return 1;
}

int importGlobalConnectivity(const char *SC_cap_filename, const char *SC_dist_filename, const char *SC_inputreg_filename, int regions,
                             float **region_activity, struct Xi_p **reg_globinp_p, float global_trans_v,
                             int **n_conn_table, float G_J_NMDA, struct SC_capS **SC_cap, float **SC_rowsums, struct SC_inpregS **SC_inpreg)
{
    (void)SC_inputreg_filename;
    int i,j,k,maxdelay=0,tmpint;
    float *region_activity_p;
    double tmp,tmp2;
    struct Xi_p *reg_globinp_pp;
    struct SC_capS *SC_capp;
    struct SC_inpregS *SC_inpregp;

    FILE *file_cap=fopen(SC_cap_filename,"r");
    FILE *file_dist=fopen(SC_dist_filename,"r");
    if (!file_cap || !file_dist){ printf("\nERROR: Could not open SC files. Terminating...\n\n"); exit(EXIT_FAILURE); }

    *SC_rowsums   = (float *)_mm_malloc(regions*sizeof(float),16);
    *n_conn_table = (int   *)_mm_malloc(regions*sizeof(int),16);
    *SC_cap       = (struct SC_capS *)_mm_malloc(regions*sizeof(struct SC_capS),16);
    *SC_inpreg    = (struct SC_inpregS *)_mm_malloc(regions*sizeof(struct SC_inpregS),16);
    SC_capp    = *SC_cap;
    SC_inpregp = *SC_inpreg;
    if(!*n_conn_table || !SC_capp || !*SC_rowsums || !SC_inpregp){
        printf("Running out of memory. Terminating.\n"); if(file_dist)fclose(file_dist); if(file_cap)fclose(file_cap); exit(2);
    }

    int n_entries=0, curr_col=0, curr_row=0, curr_row_nonzero=0;
    double tmp_max=-9999;
    while(fscanf(file_dist,"%lf",&tmp)!=EOF && fscanf(file_cap,"%lf",&tmp2)!=EOF){
        if(tmp_max<tmp) tmp_max = tmp;
        n_entries++;
        if (tmp2>0.0) curr_row_nonzero++;
        curr_col++;
        if(curr_col==regions){
            curr_col=0;
            (*n_conn_table)[curr_row] = curr_row_nonzero;
            curr_row_nonzero = 0;
            curr_row++;
        }
    }
    if (n_entries < regions*regions){ printf("ERROR: SC file too short.\n"); exit(EXIT_FAILURE); }
    rewind(file_dist); rewind(file_cap);

    maxdelay = (int)(((tmp_max/global_trans_v)*10)+0.5);
    if(maxdelay<1) maxdelay=1;

    *region_activity = (float *)_mm_malloc(maxdelay*regions*sizeof(float),16);
    region_activity_p = *region_activity;
    *reg_globinp_p = (struct Xi_p *)_mm_malloc(maxdelay*regions*sizeof(struct Xi_p),16);
    reg_globinp_pp = *reg_globinp_p;
    if(!region_activity_p || !reg_globinp_p){ printf("Running out of memory. Terminating.\n"); fclose(file_dist); exit(2); }
    for (j=0;j<maxdelay*regions;j++) region_activity_p[j]=0.001f;

    int ring_buff_position;
    for (i=0;i<regions;i++){
        if((*n_conn_table)[i] > 0){
            SC_capp[i].cap       = (float*)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_inpregp[i].inpreg = (int  *)_mm_malloc(((*n_conn_table)[i])*sizeof(int)  ,16);
            if(!SC_capp[i].cap || !SC_inpregp[i].inpreg){ printf("Running out of memory. Terminating.\n"); exit(2); }

            for (j=0;j<maxdelay;j++){
                reg_globinp_pp[i+j*regions].Xi_elems = (float **)_mm_malloc(((*n_conn_table)[i])*sizeof(float *),16);
                if(!reg_globinp_pp[i+j*regions].Xi_elems){ printf("Running out of memory. Terminating.\n"); exit(2); }
            }

            float sum_caps=0.0f;
            curr_row_nonzero=0;
            for (j=0;j<regions;j++){
                if (fscanf(file_cap,"%lf",&tmp)!=EOF && fscanf(file_dist,"%lf",&tmp2)!=EOF){
                    if (tmp>0.0){
                        tmpint = (int)(((tmp2/global_trans_v)*10)+0.5);
                        if (tmpint<0 || tmpint>maxdelay){ printf("\nERROR: Bad delay %d -> %d. Terminating...\n\n",i,j); exit(EXIT_FAILURE); }
                        if (tmpint<=0) tmpint=1;

                        SC_capp[i].cap[curr_row_nonzero] = (float)tmp * G_J_NMDA;
                        sum_caps += SC_capp[i].cap[curr_row_nonzero];
                        SC_inpregp[i].inpreg[curr_row_nonzero] = j;

                        ring_buff_position = maxdelay*regions - tmpint*regions + j;
                        for (k=0;k<maxdelay;k++){
                            reg_globinp_pp[i+k*regions].Xi_elems[curr_row_nonzero] = &region_activity_p[ring_buff_position];
                            ring_buff_position += regions;
                            if (ring_buff_position > (maxdelay*regions-1)) ring_buff_position -= maxdelay*regions;
                        }
                        curr_row_nonzero++;
                    }
                }else{
                    printf("\nERROR: Unexpected end-of-file in SC files. Terminating...\n\n");
                    exit(EXIT_FAILURE);
                }
            }
            if (sum_caps <= 0){ printf("\nERROR: Sum of caps <= 0 for node %d.\n",i); exit(EXIT_FAILURE); }
            (*SC_rowsums)[i] = sum_caps;
        }
    }

    fclose(file_dist); fclose(file_cap);
    return maxdelay;
}

static void initialize_thread_barriers(int n_threads){
    pthread_barrier_init(&mybarrier1, NULL, n_threads);
    pthread_barrier_init(&mybarrier2, NULL, n_threads);
    pthread_barrier_init(&mybarrier3, NULL, n_threads);
}

/* thread payload */
typedef struct _thread_data_t {
    int     tid, rand_num_seed;
    int     nodes, nodes_vec, fake_nodes, n_threads, vectorization_grade, time_steps, BOLD_TR, BOLD_ts_len;
    float   *J_i;
    int     reg_act_size;
    float   *region_activity;
    float   model_dt;
    __m128  _gamma,_one,_imintau_E,_dt,_sigma_sqrt_dt,_sigma,_gamma_I,_imintau_I,_min_d_I,_b_I,_J_NMDA,_w_I__I_0,_a_I,_min_d_E,_b_E,_a_E,_w_plus_J_NMDA,_w_E__I_0;
    struct SC_capS *SC_cap;
    struct SC_inpregS *SC_inpreg;
    struct Xi_p *reg_globinp_p;
    int     *n_conn_table;
    float   *BOLD_ex;
    const char    *output_BOLD_file;
    float   *CBV_ex;
    const char    *output_CBV_file;

    // electrical streaming
    int     ELEC_TR;
    int     ELEC_ts_len;
    float   *ELEC_ex;
    const char    *output_ELEC_file;
    int     num_elec_samples;

    // NEW: stimuli (shared read-only across threads)
    Stim_t *stimuli;
    int     n_stimuli;
} thread_data_t;

void *run_simulation(void *arg)
{
    int j, i_node_vec, i_node_vec_local, k, int_i, ts;
    float tmpglobinput;
    __m128 _tmp_H_E, _tmp_H_I, _tmp_I_I, _tmp_I_E;
    float tmp_exp_E[4] __attribute__((aligned(16)));
    float tmp_exp_I[4] __attribute__((aligned(16)));
    __m128 *_tmp_exp_E = (__m128*)tmp_exp_E;
    __m128 *_tmp_exp_I = (__m128*)tmp_exp_I;
    int ring_buf_pos = 0;

    thread_data_t *thr_data = (thread_data_t *)arg;
    int t_id = thr_data->tid;
    int nodes = thr_data->nodes;
    int fake_nodes = thr_data->fake_nodes;
    const int nodes_vec = thr_data->nodes_vec;
    int n_threads = thr_data->n_threads;
    int vectorization_grade = thr_data->vectorization_grade;
    int reg_act_size = thr_data->reg_act_size;
    int *n_conn_table = thr_data->n_conn_table;
    float *J_i = thr_data->J_i;
    float *region_activity = thr_data->region_activity;
    float *BOLD_ex = thr_data->BOLD_ex;
    float *CBV_ex = thr_data->CBV_ex;
    const __m128 _gamma         = thr_data->_gamma;
    const __m128 _one           = thr_data->_one;
    const __m128 _imintau_E     = thr_data->_imintau_E;
    const __m128 _dt            = thr_data->_dt;
    const __m128 _sigma_sqrt_dt = thr_data->_sigma_sqrt_dt;
    const __m128 _imintau_I     = thr_data->_imintau_I;
    const __m128 _min_d_I       = thr_data->_min_d_I;
    const __m128 _b_I           = thr_data->_b_I;
    const __m128 _J_NMDA        = thr_data->_J_NMDA;
    const __m128 _w_I__I_0      = thr_data->_w_I__I_0;
    const __m128 _min_d_E       = thr_data->_min_d_E;
    const __m128 _b_E           = thr_data->_b_E;
    const __m128 _w_plus_J_NMDA = thr_data->_w_plus_J_NMDA;
    const __m128 _w_E__I_0      = thr_data->_w_E__I_0;
    struct SC_capS *SC_cap      = thr_data->SC_cap;
    int time_steps              = thr_data->time_steps;
    int BOLD_TR                 = thr_data->BOLD_TR;
    float model_dt              = thr_data->model_dt;
    int BOLD_ts_len             = thr_data->BOLD_ts_len;

    // electrical
    int ELEC_TR    = thr_data->ELEC_TR;
    int ELEC_ts_len= thr_data->ELEC_ts_len;
    float *ELEC_ex = thr_data->ELEC_ex;
    
    // stimuli
    Stim_t *stimuli  = thr_data->stimuli;
    int     n_stimuli = thr_data->n_stimuli;
    
    // per-thread ranges
    int nodes_vec_mt     = divRoundUp(nodes_vec, n_threads);
    int nodes_mt         = nodes_vec_mt * vectorization_grade;

    int start_nodes_vec_mt = t_id       * nodes_vec_mt;
    int start_nodes_mt     = t_id       * nodes_mt;
    int end_nodes_vec_mt   = (t_id + 1) * nodes_vec_mt;
    int end_nodes_mt       = (t_id + 1) * nodes_mt;
    int end_nodes_mt_glob  = end_nodes_mt;

    if (end_nodes_mt > nodes){
        end_nodes_vec_mt  = nodes_vec;
        end_nodes_mt      = fake_nodes;
        end_nodes_mt_glob = nodes; // last thread real end
        nodes_mt          = end_nodes_mt - start_nodes_mt;
        nodes_vec_mt      = end_nodes_vec_mt - start_nodes_vec_mt;
    }

    const int nodes_real = end_nodes_mt_glob - start_nodes_mt; // real nodes owned by this thread

    // Allocate an aligned buffer for our random numbers
    int rand_buf_size = 10 * nodes_vec_mt * 8;
    float *rand_buf = (float *)_mm_malloc(rand_buf_size * sizeof(float), 16);
    
    if (!rand_buf) {
        printf("ERROR: Out of memory allocating random buffer.\n"); 
        exit(1); 
    }

    printf("thread %d: start: %d end: %d size: %d\n", t_id, start_nodes_mt, end_nodes_mt, nodes_mt);

    if (nodes_vec_mt <= 1){
        printf("Ineffective splitting. Terminating.\n");
        exit(EXIT_FAILURE);
    }

    if (t_id == 0) initialize_thread_barriers(n_threads);
    pthread_barrier_wait(&mybarrier_base);

    gsl_rng *r = gsl_rng_alloc (gsl_rng_mt19937);
    gsl_rng_set (r, (t_id + thr_data->rand_num_seed));
    srand((unsigned)(t_id + thr_data->rand_num_seed));

    float *meanFR           = (float *)_mm_malloc(nodes_mt*sizeof(float),16);
    float *meanFR_INH       = (float *)_mm_malloc(nodes_mt*sizeof(float),16);
    float *global_input     = (float *)_mm_malloc(nodes_mt*sizeof(float),16);
    float *global_input_FFI = (float *)_mm_malloc(nodes_mt*sizeof(float),16);
    float *S_i_E            = (float *)_mm_malloc(nodes_mt*sizeof(float),16);
    float *S_i_I            = (float *)_mm_malloc(nodes_mt*sizeof(float),16);
    float *J_i_local        = (float *)_mm_malloc(nodes_mt*sizeof(float),16);
    float *stim_E           = (float *)_mm_malloc(nodes_mt*sizeof(float),16);

    if(!meanFR || !meanFR_INH || !global_input || !global_input_FFI || !S_i_E || !S_i_I || !J_i_local || !stim_E){
        printf("ERROR: Running out of memory. Aborting...\n"); exit(EXIT_FAILURE);
    }

    __m128 *_meanFR         = (__m128*)meanFR;
    __m128 *_meanFR_INH     = (__m128*)meanFR_INH;
    __m128 *_global_input   = (__m128*)global_input;
    __m128 *_global_input_FFI = (__m128*)global_input_FFI;
    __m128 *_S_i_E          = (__m128*)S_i_E;
    __m128 *_S_i_I          = (__m128*)S_i_I;
    __m128 *_J_i_local      = (__m128*)J_i_local;
    __m128 *_stim_E         = (__m128*)stim_E;

    // Balloon-Windkessel
    float rho=0.34f, alpha=0.32f, tau=0.98f, y=1.0f/0.41f, kappa=1.0f/0.65f;
    float V_0=0.02f, k1=7*rho, k2=2.0f, k3=2*rho-0.2f, oneminrho=(1.0f - rho);
    float f_tmp;
    float *BOLD     = (float *)_mm_malloc(nodes_mt * BOLD_ts_len * sizeof(float),16);
    float *CBV      = (float *)_mm_malloc(nodes_mt * BOLD_ts_len * sizeof(float),16);
    float *ELEC     = (float *)_mm_malloc(nodes_mt * ELEC_ts_len * sizeof(float),16);

    float *bw_x_ex  = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float *bw_f_ex  = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float *bw_nu_ex = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float *bw_q_ex  = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    if(!BOLD || !CBV || !ELEC || !bw_x_ex || !bw_f_ex || !bw_nu_ex || !bw_q_ex){ printf("ERROR: Running out of memory. Aborting...\n"); exit(EXIT_FAILURE); }

    // reset arrays
    ring_buf_pos = 0;
    for (j=0;j<nodes_mt;j++){
        meanFR[j]=0.0f; meanFR_INH[j]=0.0f;
        global_input[j]=0.0f; global_input_FFI[j]=0.0f;
        S_i_E[j]=0.0f; S_i_I[j]=0.0f;
        J_i_local[j]= J_i[j + start_nodes_mt];
        stim_E[j]=0.0f;   // NEW
    }
    if (t_id==0){
        for (j=0;j<thr_data->reg_act_size;j++) region_activity[j]=0.0f;
    }
    for (j=0;j<nodes_mt;j++){
        bw_x_ex[j]=0.0f; bw_f_ex[j]=1.0f; bw_nu_ex[j]=1.0f; bw_q_ex[j]=1.0f;
    }
    for (j=0;j<nodes_mt * BOLD_ts_len; j++){   // NEW
        CBV[j] = 0.0f;
    }
    pthread_barrier_wait(&mybarrier1);

    int ts_bold_i = 0;
    int ts_elec_i = 0;

    for (ts=0; ts<time_steps; ts++){
        if (t_id==0) printf("%.1f %% \r", ((float)ts / (float)time_steps) * 100.0f );
        
        // build external excitatory stimulus for this time step
        for (j = 0; j < nodes_mt; j++) stim_E[j] = 0.0f;

        for (int s = 0; s < n_stimuli; s++) {
            if (ts >= stimuli[s].start_ts && ts < stimuli[s].end_ts) {
                int node = stimuli[s].node;
                if (node >= start_nodes_mt && node < end_nodes_mt_glob) {
                    int local = node - start_nodes_mt;   // 0 .. nodes_real-1
                    if (local >= 0 && local < nodes_real) {
                        stim_E[local] += stimuli[s].amplitude;
                    }
                }
            }
        }

        // Pre-generate all random numbers for this entire 1 ms step
        for (int i_rnd = 0; i_rnd < rand_buf_size; i_rnd++) {
            rand_buf[i_rnd] = (float)gsl_ran_gaussian(r, 1.0);
        }
        
        int rand_idx = 0; // Pointer to track where we are in the buffer

        // 10 sub-steps per ms (dt=0.1 ms)
        for (int_i=0; int_i<10; int_i++){
            pthread_barrier_wait(&mybarrier2);

            // global coupling
            i_node_vec_local = 0;
            for (j=start_nodes_mt; j<end_nodes_mt_glob; j++){
                tmpglobinput=0.0f;
                for (k=0;k<n_conn_table[j];k++){
                    tmpglobinput += *thr_data->reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap[j].cap[k];
                }
                global_input[i_node_vec_local] = tmpglobinput;
                i_node_vec_local++;
            }

            // vectorized local dynamics
            // vectorized local dynamics
            i_node_vec_local = 0;
            for (i_node_vec=start_nodes_vec_mt; i_node_vec<end_nodes_vec_mt; i_node_vec++){
                __m128 _stim_vec = _stim_E[i_node_vec_local];

                _tmp_I_E = _mm_sub_ps(
                    _mm_mul_ps(
                        thr_data->_a_E,
                        _mm_add_ps(
                            _mm_add_ps(_w_E__I_0,
                                       _mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec_local])),
                            
                                       // global input + external stimulus - inhibitory feedback
                            _mm_add_ps(
                                _mm_sub_ps(_global_input[i_node_vec_local],
                                           _mm_mul_ps(_J_i_local[i_node_vec_local], _S_i_I[i_node_vec_local])),
                                _stim_vec
                            )
                        )
                    ),
                    _b_E
                );
                *_tmp_exp_E = _mm_mul_ps(_min_d_E, _tmp_I_E);
                tmp_exp_E[0] = tmp_exp_E[0] ? expf(tmp_exp_E[0]) : 0.9f;
                tmp_exp_E[1] = tmp_exp_E[1] ? expf(tmp_exp_E[1]) : 0.9f;
                tmp_exp_E[2] = tmp_exp_E[2] ? expf(tmp_exp_E[2]) : 0.9f;
                tmp_exp_E[3] = tmp_exp_E[3] ? expf(tmp_exp_E[3]) : 0.9f;
                _tmp_H_E = _mm_div_ps(_tmp_I_E, _mm_sub_ps(_one, *_tmp_exp_E));
                _meanFR[i_node_vec_local] = _mm_add_ps(_meanFR[i_node_vec_local], _tmp_H_E);

                _tmp_I_I = _mm_sub_ps(_mm_mul_ps(thr_data->_a_I,_mm_sub_ps(_mm_add_ps(_mm_add_ps(_w_I__I_0,_global_input_FFI[i_node_vec_local]),
                                   _mm_mul_ps(_J_NMDA,_S_i_E[i_node_vec_local])), _S_i_I[i_node_vec_local])), _b_I);
                *_tmp_exp_I = _mm_mul_ps(_min_d_I, _tmp_I_I);
                tmp_exp_I[0] = tmp_exp_I[0] ? expf(tmp_exp_I[0]) : 0.9f;
                tmp_exp_I[1] = tmp_exp_I[1] ? expf(tmp_exp_I[1]) : 0.9f;
                tmp_exp_I[2] = tmp_exp_I[2] ? expf(tmp_exp_I[2]) : 0.9f;
                tmp_exp_I[3] = tmp_exp_I[3] ? expf(tmp_exp_I[3]) : 0.9f;
                _tmp_H_I = _mm_div_ps(_tmp_I_I, _mm_sub_ps(_one, *_tmp_exp_I));
                _meanFR_INH[i_node_vec_local] = _mm_add_ps(_meanFR_INH[i_node_vec_local], _tmp_H_I);

                __m128 _rand_I = _mm_load_ps(&rand_buf[rand_idx]);
                rand_idx += 4;
                _S_i_I[i_node_vec_local] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma_sqrt_dt, _rand_I), _S_i_I[i_node_vec_local]),
                                                      _mm_mul_ps(_dt, _mm_add_ps(_mm_mul_ps(_imintau_I, _S_i_I[i_node_vec_local]),
                                                                                 _mm_mul_ps(_tmp_H_I, thr_data->_gamma_I))));

                __m128 _rand_E = _mm_load_ps(&rand_buf[rand_idx]);
                rand_idx += 4;
                _S_i_E[i_node_vec_local] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma_sqrt_dt, _rand_E), _S_i_E[i_node_vec_local]),
                                                      _mm_mul_ps(_dt, _mm_add_ps(_mm_mul_ps(_imintau_E, _S_i_E[i_node_vec_local]),
                                                                                 _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_one,_S_i_E[i_node_vec_local]), _gamma),
                                                                                            _tmp_H_E))));
                i_node_vec_local++;
            }

            // clamp real nodes only
            for (j=0;j<nodes_real;j++){
                S_i_E[j] = (S_i_E[j]<0.0f)?0.0f:((S_i_E[j]>1.0f)?1.0f:S_i_E[j]);
                S_i_I[j] = (S_i_I[j]<0.0f)?0.0f:((S_i_I[j]>1.0f)?1.0f:S_i_I[j]);
            }

            // copy E to shared ring buffer (real nodes only) with bounds check
            if ((ring_buf_pos + start_nodes_mt + nodes_real) > reg_act_size){
                fprintf(stderr,"ERROR: ring buffer overflow (thread %d): rb=%d start=%d real=%d cap=%d\n",
                        t_id, ring_buf_pos, start_nodes_mt, nodes_real, reg_act_size);
                exit(EXIT_FAILURE);
            }
            memcpy(&region_activity[ring_buf_pos + start_nodes_mt], S_i_E, (size_t)nodes_real * sizeof(float));

            // advance ring buffer head by 'nodes'
            ring_buf_pos = (ring_buf_pos < (reg_act_size - nodes)) ? (ring_buf_pos + nodes) : 0;
        }

        // BW model update and BOLD sample for real nodes
        for (j=0;j<nodes_real;j++){
            bw_x_ex[j]  = bw_x_ex[j]  +  model_dt * (S_i_E[j] - kappa * bw_x_ex[j] - y * (bw_f_ex[j] - 1.0f));
            f_tmp       = bw_f_ex[j]  +  model_dt * bw_x_ex[j];
            bw_nu_ex[j] = bw_nu_ex[j] +  model_dt * (1.0f/tau) * (bw_f_ex[j] - powf(bw_nu_ex[j], 1.0f/alpha));
            bw_q_ex[j]  = bw_q_ex[j]  +  model_dt * (1.0f/tau) * (bw_f_ex[j]*(1.0f - powf(oneminrho,(1.0f/bw_f_ex[j])))/rho
                                         - powf(bw_nu_ex[j],1.0f/alpha) * bw_q_ex[j] / bw_nu_ex[j]);
            bw_f_ex[j]  = f_tmp;
        }

        if (ts % BOLD_TR == 0){
            for (j=0;j<nodes_real;j++){
                int idx = ts_bold_i + j * BOLD_ts_len;
                BOLD[idx] = 100.0f / rho * V_0 *
                    (k1 * (1.0f - bw_q_ex[j]) + k2 * (1.0f - bw_q_ex[j]/bw_nu_ex[j]) + k3 * (1.0f - bw_nu_ex[j]));
                CBV[idx]  = bw_nu_ex[j];
            }
            ts_bold_i++;
        }

        // Electrical sample -> stream thread block (includes padding width nodes_mt by design)
        // Electrical sample -> store in memory
        if (ts % ELEC_TR == 0){
            for (j = 0; j < nodes_real; j++) {
                ELEC[ts_elec_i + j * ELEC_ts_len] = S_i_E[j];
            }
            ts_elec_i++;
        }
    }

    thr_data->num_elec_samples = ts_elec_i;

    // Copy BOLD back only for real nodes
    memcpy(&BOLD_ex[start_nodes_mt * BOLD_ts_len], BOLD, (size_t)nodes_real * BOLD_ts_len * sizeof(float));
    memcpy(&CBV_ex[start_nodes_mt * BOLD_ts_len],  CBV,  (size_t)nodes_real * BOLD_ts_len * sizeof(float));
    memcpy(&ELEC_ex[start_nodes_mt * ELEC_ts_len], ELEC, (size_t)nodes_real * ELEC_ts_len * sizeof(float));

    pthread_barrier_wait(&mybarrier3);

    if (t_id==0){
        FILE *FCout_BOLD = fopen(thr_data->output_BOLD_file, "w");
        if (!FCout_BOLD){ printf("ERROR: Could not open BOLD output file.\n"); }
        else{
            for (j=0;j<nodes;j++){
                for (k=0;k<ts_bold_i;k++){
                    fprintf(FCout_BOLD, "%.5f ", BOLD_ex[j*BOLD_ts_len + k]);
                }
                fprintf(FCout_BOLD, "\n");
            }
            fclose(FCout_BOLD);
        }
        FILE *FCout_CBV = fopen(thr_data->output_CBV_file, "w");
        if (!FCout_CBV){
            printf("ERROR: Could not open CBV output file.\n");
        } else {
            for (j=0;j<nodes;j++){
                for (k=0;k<ts_bold_i;k++){
                    fprintf(FCout_CBV, "%.5f ", CBV_ex[j*BOLD_ts_len + k]);
                }
                fprintf(FCout_CBV, "\n");
            }
            fclose(FCout_CBV);
        }
        FILE *FCout_ELEC = fopen(thr_data->output_ELEC_file, "w");
        if (!FCout_ELEC){
            printf("ERROR: Could not open Electrical output file.\n");
        } else {
            for (j=0; j<nodes; j++){
                for (k=0; k<ts_elec_i; k++){
                    fprintf(FCout_ELEC, "%.5f ", ELEC_ex[j*ELEC_ts_len + k]);
                }
                fprintf(FCout_ELEC, "\n");
            }
            fclose(FCout_ELEC);
        }
    }

    // --- MEMORY CLEANUP ---
    _mm_free(meanFR);
    _mm_free(meanFR_INH);
    _mm_free(global_input);
    _mm_free(global_input_FFI);
    _mm_free(S_i_E);
    _mm_free(S_i_I);
    _mm_free(J_i_local);
    _mm_free(stim_E);
    _mm_free(BOLD);
    _mm_free(CBV);
    _mm_free(ELEC);
    _mm_free(bw_x_ex);
    _mm_free(bw_f_ex);
    _mm_free(bw_nu_ex);
    _mm_free(bw_q_ex);

    _mm_free(rand_buf);
    gsl_rng_free(r);
    pthread_exit(NULL);
}

/*
 Usage: tvb <#threads>
*/
int main(int argc, char *argv[])
{
    if (argc != 2 || atoi(argv[1]) <= 0){
        printf("\nERROR: Invalid arguments.\n\nUsage: tvbii <#threads>\n\n");
        for (int i=0;i<argc;i++) printf("%s\n", argv[i]);
        exit(EXIT_FAILURE);
    }

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    int i,j;
    int n_threads = atoi(argv[1]);

    // fixed values
    const float dt = 0.1f;
    const float sqrt_dt = sqrtf(dt);
    const float model_dt = 0.001f;
    const int vectorization_grade = 4;

    // these values are just a reference, the ones used in the simulation are always read from the disk
    int   time_steps     = (int)(667*1.94*1000);
    int   nodes          = 84;
    int   fake_nodes     = 84;
    float global_trans_v = 1.0f;
    float G              = 0.5f;
    int   BOLD_TR        = 1940;
    int   rand_num_seed  = 1403;
    float w_plus=1.4f, J_NMDA=0.15f;
    float tmpJi=1.0f;

    // nmm parameters
    const float a_E=310.0f, b_E=125.0f, d_E=0.16f;
    const float a_I=615.0f, b_I=177.0f, d_I=0.087f;
    const float gamma=0.641f/1000.0f;
    const float tau_E=100.0f, tau_I=10.0f;
    float sigma=0.00316228f;
    const float I_0=0.182f, w_E=1.0f, w_I=0.7f, gamma_I=1.0f/1000.0f;

    // input and output filenames
    const char *param_file = "/input/param_set.txt";
    const char *stim_file = "/input/stimulus.txt";
    const char *cap_file = "/input/SC_weights.txt";
    const char *dist_file = "/input/SC_distances.txt";
    const char *reg_file = "/input/SC_regionids.txt";
    const char *output_BOLD_file = "/output/BOLD.txt";
    const char *output_CBV_file = "/output/CBV.txt";
    const char *output_ELEC_file = "/output/ELEC.txt";


    // read params
    FILE *file;
    file=fopen(param_file,"r");
    if (!file){ printf("\nERROR: Could not open file %s.\n", param_file); exit(EXIT_FAILURE); }
    if(!(fscanf(file,"%d",&nodes)!=EOF && fscanf(file,"%f",&G)!=EOF && fscanf(file,"%f",&J_NMDA)!=EOF &&
         fscanf(file,"%f",&w_plus)!=EOF && fscanf(file,"%f",&tmpJi)!=EOF && fscanf(file,"%f",&sigma)!=EOF &&
         fscanf(file,"%d",&time_steps)!=EOF && fscanf(file,"%d",&BOLD_TR)!=EOF && fscanf(file,"%f",&global_trans_v)!=EOF &&
         fscanf(file,"%d",&rand_num_seed)!=EOF)){
        printf("\nERROR: Unexpected end-of-file in %s\n", param_file); exit(EXIT_FAILURE);
    }
    fclose(file);
    
    if (nodes % vectorization_grade != 0){
        printf("\nWarning: nodes (%d) not multiple of SIMD width (%d). Adding fake nodes...\n\n", nodes, vectorization_grade);
        int remainder = nodes % vectorization_grade;
        if (remainder > 0) fake_nodes = nodes + (vectorization_grade - remainder);
    }else{
        fake_nodes = nodes;
    }

    srand((unsigned)rand_num_seed);

    const float sigma_sqrt_dt = sqrt_dt * sigma;
    const int   nodes_vec     = fake_nodes / vectorization_grade;
    const float min_d_E       = -1.0f * d_E;
    const float min_d_I       = -1.0f * d_I;
    const float imintau_E     = -1.0f / tau_E;
    const float imintau_I     = -1.0f / tau_I;
    const float w_E__I_0      = w_E * I_0;
    const float w_I__I_0      = w_I * I_0;
    const float one           = 1.0f;
    const float w_plus__J_NMDA= w_plus * J_NMDA;
    const float G_J_NMDA      = G * J_NMDA;
          float TR            = (float)BOLD_TR / 1000.0f;
          int   BOLD_ts_len   = (int)(time_steps / (TR / model_dt) + 1);

    // electrical downsampling (ms)
    int ELEC_TR = 1; // adjust if needed
    int ELEC_ts_len = time_steps / ELEC_TR + 1;

    // load optional stimuli
    Stim_t *stimuli = NULL;
    int     n_stimuli = 0;
    load_stimuli(stim_file, nodes, time_steps, &stimuli, &n_stimuli);
    if (n_stimuli > 0) {
        printf("Loaded %d stimuli from %s\n", n_stimuli, stim_file);
    }
    // import connectivity
    int         *n_conn_table;
    float       *region_activity, *SC_rowsums;
    struct Xi_p *reg_globinp_p;
    struct SC_capS    *SC_cap;
    struct SC_inpregS *SC_inpreg;

    int maxdelay = importGlobalConnectivity(cap_file, dist_file, reg_file, nodes,
                                            &region_activity, &reg_globinp_p, global_trans_v,
                                            &n_conn_table, G_J_NMDA, &SC_cap, &SC_rowsums, &SC_inpreg);
    int reg_act_size = nodes * maxdelay;

    float *J_i     = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *BOLD_ex = (float *)_mm_malloc(nodes * BOLD_ts_len * sizeof(float),16);
    float *CBV_ex  = (float *)_mm_malloc(nodes * BOLD_ts_len * sizeof(float),16);
    float *ELEC_ex = (float *)_mm_malloc(nodes * ELEC_ts_len * sizeof(float),16);

    if(!J_i || !BOLD_ex || !CBV_ex){
            printf("ERROR: Running out of memory.\n");
            if (J_i)    _mm_free(J_i);
            if (BOLD_ex)_mm_free(BOLD_ex);
            if (CBV_ex) _mm_free(CBV_ex);   // NEW
            return EXIT_FAILURE;
        }  
    for (j=0;j<fake_nodes;j++) J_i[j]=tmpJi;

    const __m128 _dt                 = _mm_load1_ps(&dt);
    const __m128 _sigma_sqrt_dt_v    = _mm_load1_ps(&sigma_sqrt_dt);
    const __m128 _w_plus_J_NMDA      = _mm_load1_ps(&w_plus__J_NMDA);
    const __m128 _a_E                = _mm_load1_ps(&a_E);
    const __m128 _b_E                = _mm_load1_ps(&b_E);
    const __m128 _min_d_E            = _mm_load1_ps(&min_d_E);
    const __m128 _a_I                = _mm_load1_ps(&a_I);
    const __m128 _b_I                = _mm_load1_ps(&b_I);
    const __m128 _min_d_I            = _mm_load1_ps(&min_d_I);
    const __m128 _gamma              = _mm_load1_ps(&gamma);
    const __m128 _gamma_I_v          = _mm_load1_ps(&gamma_I);
    const __m128 _imintau_E_v        = _mm_load1_ps(&imintau_E);
    const __m128 _imintau_I_v        = _mm_load1_ps(&imintau_I);
    const __m128 _w_E__I_0_v         = _mm_load1_ps(&w_E__I_0);
    const __m128 _w_I__I_0_v         = _mm_load1_ps(&w_I__I_0);
    float tmp_sigma = sigma * dt;
    const __m128 _sigma              = _mm_load1_ps(&tmp_sigma);
    const __m128 _one                = _mm_load1_ps(&one);
    const __m128 _J_NMDA_v           = _mm_load1_ps(&J_NMDA);

    // threading
    pthread_barrier_init(&mybarrier_base, NULL, n_threads);

    pthread_t *thr = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
    thread_data_t *thr_data = (thread_data_t *)malloc((size_t)n_threads * sizeof(thread_data_t));
    if (!thr || !thr_data) {
        printf("ERROR: Out of memory allocating thread arrays.\n");
        exit(1); // Remember our new exit codes!
    }
    
    int rc;

    for (i=0;i<n_threads;i++){
        memset(&thr_data[i],0,sizeof(thread_data_t));
        thr_data[i].tid                 = i;
        thr_data[i].n_threads           = n_threads;
        thr_data[i].vectorization_grade = vectorization_grade;
        thr_data[i].nodes               = nodes;
        thr_data[i].fake_nodes          = fake_nodes;
        thr_data[i].J_i                 = J_i;
        thr_data[i].reg_act_size        = reg_act_size;
        thr_data[i].region_activity     = region_activity;
        thr_data[i]._gamma              = _gamma;
        thr_data[i]._one                = _one;
        thr_data[i]._imintau_E          = _imintau_E_v;
        thr_data[i]._dt                 = _dt;
        thr_data[i]._sigma_sqrt_dt      = _sigma_sqrt_dt_v;
        thr_data[i]._sigma              = _sigma;
        thr_data[i]._gamma_I            = _gamma_I_v;
        thr_data[i]._imintau_I          = _imintau_I_v;
        thr_data[i]._min_d_I            = _min_d_I;
        thr_data[i]._b_I                = _b_I;
        thr_data[i]._J_NMDA             = _J_NMDA_v;
        thr_data[i]._w_I__I_0           = _w_I__I_0_v;
        thr_data[i]._a_I                = _a_I;
        thr_data[i]._min_d_E            = _min_d_E;
        thr_data[i]._b_E                = _b_E;
        thr_data[i]._a_E                = _a_E;
        thr_data[i]._w_plus_J_NMDA      = _w_plus_J_NMDA;
        thr_data[i]._w_E__I_0           = _w_E__I_0_v;
        thr_data[i].nodes_vec           = nodes_vec;
        thr_data[i].SC_cap              = SC_cap;
        thr_data[i].SC_inpreg           = SC_inpreg;
        thr_data[i].reg_globinp_p       = reg_globinp_p;
        thr_data[i].n_conn_table        = n_conn_table;
        thr_data[i].time_steps          = time_steps;
        thr_data[i].BOLD_TR             = BOLD_TR;
        thr_data[i].model_dt            = model_dt;
        thr_data[i].BOLD_ex             = BOLD_ex;
        thr_data[i].rand_num_seed       = rand_num_seed;
        thr_data[i].BOLD_ts_len         = BOLD_ts_len;
        thr_data[i].output_BOLD_file         = output_BOLD_file;
        thr_data[i].CBV_ex              = CBV_ex;
        thr_data[i].output_CBV_file     = output_CBV_file;

        // stimuli (same pointer for all threads, read-only)
        thr_data[i].stimuli             = stimuli;
        thr_data[i].n_stimuli           = n_stimuli;

        // electrical streaming
        thr_data[i].ELEC_TR             = ELEC_TR;
        thr_data[i].ELEC_ts_len         = ELEC_ts_len;
        thr_data[i].ELEC_ex = ELEC_ex;
        thr_data[i].output_ELEC_file    = output_ELEC_file;

        rc = pthread_create(&thr[i], NULL, run_simulation, &thr_data[i]);
        if (rc){ fprintf(stderr,"error: pthread_create, rc: %d\n", rc); return EXIT_FAILURE; }
    }

    for (i=0;i<n_threads;i++) pthread_join(thr[i], NULL);
    printf("Threads finished. Back to main thread.\n");

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double wall_time_used = (end_time.tv_sec - start_time.tv_sec) + 
                            (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    printf("Simulation finished. Execution took %.3f s for %d nodes. Goodbye!\n", wall_time_used, nodes);

    _mm_free(BOLD_ex);
    _mm_free(J_i);
    _mm_free(CBV_ex);
    _mm_free(ELEC_ex);
    if (stimuli) free(stimuli);
    free(thr);
    free(thr_data);

    // --- FINAL CLEANUP OF GLOBAL CONNECTIVITY ---
    for (i = 0; i < nodes; i++) {
        if (n_conn_table[i] > 0) {
            // Free the inner arrays for this node
            _mm_free(SC_cap[i].cap);
            _mm_free(SC_inpreg[i].inpreg);
            
            // Free the delay pointers
            for (j = 0; j < maxdelay; j++) {
                _mm_free(reg_globinp_p[i + j * nodes].Xi_elems);
            }
        }
    }
    // Free the outer arrays 
    _mm_free(SC_cap);
    _mm_free(SC_inpreg);
    _mm_free(n_conn_table);
    _mm_free(SC_rowsums);
    _mm_free(reg_globinp_p);
    _mm_free(region_activity);

    return 0;
}
