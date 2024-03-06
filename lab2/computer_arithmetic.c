#include <stdio.h>
#include <gsl/gsl_ieee_utils.h>

int main() {
    float f = 1e-33;
    int i = 1;
    while(f>0){
        printf("%i. ",i); gsl_ieee_printf_float(&f);
        printf("\n");
        f/=2.0;
        i++;
    }
    return 0;
}