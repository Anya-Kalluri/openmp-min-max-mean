#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <limits.h>
#include <time.h>

#define ARRAY_SIZE 1000000000  // 10^9 elements (close to 2^34 as mentioned)
#define MAX_VALUE 1000000000   // 10^9 domain
#define NUM_RUNS 5

// Function to get current time in microseconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Generate random data with uniform distribution
void generate_data(int *data, long long size) {
    printf("Generating %lld random numbers...\n", size);
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num();
        #pragma omp for
        for (long long i = 0; i < size; i++) {
            data[i] = rand_r(&seed) % (MAX_VALUE + 1);
        }
    }
    printf("Data generation completed.\n");
}

// Serial implementation
void serial_min_max_mean(int *data, long long size, int *min_val, int *max_val, double *mean_val) {
    *min_val = INT_MAX;
    *max_val = INT_MIN;
    long long sum = 0;
    
    for (long long i = 0; i < size; i++) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
        sum += data[i];
    }
    
    *mean_val = (double)sum / size;
}

// Parallel implementation
void parallel_min_max_mean(int *data, long long size, int *min_val, int *max_val, double *mean_val) {
    int local_min = INT_MAX;
    int local_max = INT_MIN;
    long long sum = 0;
    
    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) reduction(+:sum)
    for (long long i = 0; i < size; i++) {
        if (data[i] < local_min) local_min = data[i];
        if (data[i] > local_max) local_max = data[i];
        sum += data[i];
    }
    
    *min_val = local_min;
    *max_val = local_max;
    *mean_val = (double)sum / size;
}

int main() {
    printf("Min-Max-Mean Parallel Computing with OpenMP\n");
    printf("Array size: %d elements\n", ARRAY_SIZE);
    printf("Domain: {0, 1, ..., %d}\n\n", MAX_VALUE);
    
    // Allocate memory
    int *data = (int*)malloc(ARRAY_SIZE * sizeof(int));
    if (data == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // Generate test data
    generate_data(data, ARRAY_SIZE);
    
    // Open results file
    FILE *results_file = fopen("results.csv", "w");
    fprintf(results_file, "Threads,Runtime(s),Speedup,Min,Max,Mean\n");
    
    printf("Thread Count | Runtime(s) | Speedup | Min Value | Max Value | Mean Value\n");
    printf("-------------|------------|---------|-----------|-----------|------------\n");
    
    double serial_time = 0;
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};
    int num_thread_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    for (int t = 0; t < num_thread_configs; t++) {
        int num_threads = thread_counts[t];
        omp_set_num_threads(num_threads);
        
        double total_time = 0;
        int final_min, final_max;
        double final_mean;
        
        // Run multiple times for averaging
        for (int run = 0; run < NUM_RUNS; run++) {
            int min_val, max_val;
            double mean_val;
            
            double start_time = get_time();
            
            if (num_threads == 1) {
                serial_min_max_mean(data, ARRAY_SIZE, &min_val, &max_val, &mean_val);
            } else {
                parallel_min_max_mean(data, ARRAY_SIZE, &min_val, &max_val, &mean_val);
            }
            
            double end_time = get_time();
            total_time += (end_time - start_time);
            
            // Store results from first run
            if (run == 0) {
                final_min = min_val;
                final_max = max_val;
                final_mean = mean_val;
            }
        }
        
        double avg_time = total_time / NUM_RUNS;
        
        // Calculate speedup
        if (num_threads == 1) {
            serial_time = avg_time;
        }
        double speedup = serial_time / avg_time;
        
        // Print results
        printf("%12d | %10.4f | %7.2f | %9d | %9d | %10.2f\n", 
               num_threads, avg_time, speedup, final_min, final_max, final_mean);
        
        // Write to CSV
        fprintf(results_file, "%d,%.6f,%.4f,%d,%d,%.2f\n", 
                num_threads, avg_time, speedup, final_min, final_max, final_mean);
    }
    
    fclose(results_file);
    free(data);
    
    printf("\nResults saved to results.csv\n");
    printf("Run 'python3 plot.py' to generate graphs\n");
    
    return 0;
}