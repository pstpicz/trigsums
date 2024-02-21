#include <stdio.h>
#include <math.h>
#include "omp.h"
#include <stdlib.h>
#include "immintrin.h"
#include "trgsum.h"

double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void trg_set(int n, double *u)
{ 
  srand(1234);
  #pragma omp parallel for
  for(int i=0;i<=n;i++){
    u[i]= 1.0;  // randfrom(-1.0,1.0);
  }

}


int main(int argc, char *argv[])
{
  double *u, *v;  
  double x;

  double uc, us, vc, vs;

  double tv0,tv1,ts0,ts1, times, timev;

  int n = atoi(argv[1]);
  
  size_t size = (n+1) * sizeof(double);
  
  u = (double *)_mm_malloc(size,64);        // Allocate array on host
  v = (double *)_mm_malloc(size,64);  

  double ru, rv;

  x=atof(argv[2]);

  int p=atoi(argv[3]);  

   if(p==1){
     trg_set(n,u);
     ts0 = omp_get_wtime();
     trg_d_s_goertzel(x,n,u,&uc,&us);
     ts1 = omp_get_wtime();
     times=ts1-ts0;
   }
 
   if(p!=1){
     trg_set(n,v);
     tv0 = omp_get_wtime();
     trg_d_p_goertzel(x,n,v,&vc,&vs,p);
     tv1 = omp_get_wtime();
     timev=tv1-tv0;
   }else{
     trg_set(n,v);
     tv0 = omp_get_wtime();
     trg_d_v_goertzel(x,n,v,&vc,&vs);
     tv1 = omp_get_wtime();
     timev=tv1-tv0; 
   }
   
   

  if(p==1){
     printf("%d %d %lf %lf %lf %lf %le %le %4.2lf\n",p,n,uc,us,vc,vs,times,timev,times/timev);
   }else{
     printf("%d %d %lf %lf %le\n",p,n,vc,vs,timev);
   }


   return 0;
}


