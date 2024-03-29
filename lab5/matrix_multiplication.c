#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_blas.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <time.h>
#include <sys/times.h>
#include <sys/resource.h>

clock_t clock_start, clock_end;
struct tms st_cpu, en_cpu;

void timer_start(){ 
    clock_start = times(&st_cpu);
}
void timer_end(){ 
    clock_end = times(&en_cpu);
}

float real_time(){
    int clock_tics = sysconf(_SC_CLK_TCK);
    double real_time;
    real_time = (double ) (clock_end - clock_start)/ clock_tics;
    return real_time;
}

double **create_matrix(int n){
    double **matrix = calloc(n,sizeof(double *));
    for(int i = 0; i < n; i++){
        matrix[i] = calloc(n,sizeof(double));
        for(int j = 0; j < n; j++){
            matrix[i][j] = ((double)rand()/(double)(RAND_MAX)) * 10.0;
        }
    }
    return matrix;
}

double **create_matrix_zeros(int n){
    double **matrix = calloc(n,sizeof(double *));
    for(int i = 0; i < n; i++){
        matrix[i] = calloc(n,sizeof(double));
        for(int j = 0; j < n; j++){
            matrix[i][j] = 0.0;
        }
    }
    return matrix;
}

double *create_gsl_matrix(int n){
    double *matrix = calloc(n*n,sizeof(double));
    for(int i = 0; i < n*n; i++){
        matrix[i] = ((double)rand()/(double)(RAND_MAX)) * 10.0;
    }
    return matrix;
}

double *create_gsl_matrix_zeros(int n){
    double *matrix = calloc(n*n,sizeof(double));
    for(int i = 0; i < n*n; i++){
        matrix[i] = 0.0;
    }
    return matrix;
}

void free_matrix(double **matrix, int n){
    for(int i = 0; i < n; i++){
        free(matrix[i]);
    }
    free(matrix);
}

void naive_multiplication(double **A, double **B, double **C, int n){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k]*B[k][j];
}

void better_multiplication(double **A, double **B, double **C, int n){
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][k]*B[k][j];
}

void test(){
    FILE* file = fopen("time_test.csv","w");

    fprintf(file, "Size,Type,Time");
    double **A, **B, **C;
    float time;

    for(int i = 100; i < 1001; i+=100){
        printf("%d\n",i);
        for(int j = 0; j < 10; j++){
            A = create_matrix(i);
            B = create_matrix(i);
            C = create_matrix_zeros(i);

            timer_start();
            naive_multiplication(A,B,C,i);
            timer_end();
            time = real_time();
            fprintf(file,"\n%d,%s,%f",i,"n",time);

            free_matrix(C,i);
            C = create_matrix_zeros(i); 

            timer_start();
            better_multiplication(A,B,C,i);
            timer_end();
            time = real_time();
            fprintf(file,"\n%d,%s,%f",i,"b",time);

            double *a, *b, *c;
            a = create_gsl_matrix(i);
            b = create_gsl_matrix(i);
            c = create_gsl_matrix_zeros(i);
            gsl_matrix_view A1 = gsl_matrix_view_array(a,i,i);
            gsl_matrix_view B1 = gsl_matrix_view_array(b,i,i);
            gsl_matrix_view C1 = gsl_matrix_view_array(c,i,i);

            timer_start();
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                            1.0, &A1.matrix, &B1.matrix,
                            0.0, &C1.matrix);
            timer_end();
            time = real_time();
            fprintf(file,"\n%d,%s,%f",i,"g",time); 

            free(a);
            free(b);
            free(c);
        }
    }
    fclose(file);
}

int main(int argc, char **argv){
    srand((unsigned int)time(NULL));
    test();
    return 0;
}