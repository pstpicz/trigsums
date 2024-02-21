/** @file trgsum.c
 *  @brief Implementation of the package
 *
 * The file contains implementation of all functions.
 *
 *  @author Przemyslaw Stpiczynski
 *  @bug No know bugs.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "immintrin.h"


/**
 * @brief Auxiliary function that transposes 8x8 matrix stored columnwise in eight variables of the type _m512d
*/
__inline
void vec8x8_transpose(__m512d *col0,__m512d *col1,__m512d *col2,__m512d *col3,
                      __m512d *col4,__m512d *col5,__m512d *col6,__m512d *col7)
{


    __m512d tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    __m512d shf0, shf1, shf2, shf3, shf4, shf5, shf6, shf7;

    // Unpack and interleave double-precision floating-point elements from
    // the low and high halves of each 128-bit lane of pairs of columns
    tmp0 = _mm512_unpacklo_pd(*col0, *col1);
    tmp1 = _mm512_unpackhi_pd(*col0, *col1);
    tmp2 = _mm512_unpacklo_pd(*col2, *col3);
    tmp3 = _mm512_unpackhi_pd(*col2, *col3);
    tmp4 = _mm512_unpacklo_pd(*col4, *col5);
    tmp5 = _mm512_unpackhi_pd(*col4, *col5);
    tmp6 = _mm512_unpacklo_pd(*col6, *col7);
    tmp7 = _mm512_unpackhi_pd(*col6, *col7);


    // Shuffle 128-bits (composed of 2 double-precision floating-point elements) selected from pairs,
    // store the results (elements are copied when the corresponding mask bit is not set)
    shf0 =_mm512_mask_shuffle_f64x2(tmp0,0b11001100,tmp2,tmp2,0b10100000);
    shf1 =_mm512_mask_shuffle_f64x2(tmp1,0b11001100,tmp3,tmp3,0b10100000);
    shf2 =_mm512_mask_shuffle_f64x2(tmp2,0b00110011,tmp0,tmp0,0b11110101);
    shf3 =_mm512_mask_shuffle_f64x2(tmp3,0b00110011,tmp1,tmp1,0b11110101);
    shf4 =_mm512_mask_shuffle_f64x2(tmp4,0b11001100,tmp6,tmp6,0b10100000);
    shf5 =_mm512_mask_shuffle_f64x2(tmp5,0b11001100,tmp7,tmp7,0b10100000);
    shf6 =_mm512_mask_shuffle_f64x2(tmp6,0b00110011,tmp4,tmp4,0b11110101);
    shf7 =_mm512_mask_shuffle_f64x2(tmp7,0b00110011,tmp5,tmp5,0b11110101);

    // Shuffle 128-bits (composed of 2 double-precision floating-point elements) selected from pairs,
    // and store the results
    *col0 = _mm512_shuffle_f64x2(shf0,shf4,0b01000100);
    *col1 = _mm512_shuffle_f64x2(shf1,shf5,0b01000100);
    *col2 = _mm512_shuffle_f64x2(shf2,shf6,0b01000100);
    *col3 = _mm512_shuffle_f64x2(shf3,shf7,0b01000100);
    *col4 = _mm512_shuffle_f64x2(shf0,shf4,0b11101110);
    *col5 = _mm512_shuffle_f64x2(shf1,shf5,0b11101110);
    *col6 = _mm512_shuffle_f64x2(shf2,shf6,0b11101110);
    *col7 = _mm512_shuffle_f64x2(shf3,shf7,0b11101110);

}

/**
 * @brief Implementation of basic sequential Goertzel algorithm.
 */
void trg_d_s_goertzel(double x,int n, double *b, double *cx, double *sx)
{

    double s,s1,s2,tc,tc2;
    s1=0;
    s2=0;
    tc=cos(x);
    tc2=2.0*tc;
    for(int k=n; k>0; k--)
    {
        s=b[k]+s1*tc2-s2;
        s2=s1;
        s1=s;
    }
    *cx=b[0]+s1*tc-s2;
    *sx=s1*sin(x);
}

/**
 * @brief Implementation of vectorized Goertzel algorithm for x!=k*Pi
 */

void trg_d_v_goertzel(double x,int n,double *b, double *cx, double *sx)
{

    __m512d sk,s1,s2,tc;
    __m512d col0,col1,col2,col3,col4,col5,col6,col7;

    double z[2][8];
    double M[2][2];
    double vx[2];
    double tmp[2];

    double cosin=cos(x);
    double c=2.0*cosin;

    if(n>63) // a part of coefficients will be processed using AVX512
    {

        int n0=n-63;     // the number of coefficients excluding b_0,...,b_63
        int r=n0%64;     // the number of coefficients to be processed before the vectorized part
        int n1=n-r;      // the index of the first coefficient processed by the vectorized part
        int m=(n0-r)/8;  // the number of coefficients to be processed by the vectorized part divided by 8


        if(r==0) // no sequential computation before the vectorized part
        {
            s1=s2=_mm512_setzero_pd();
        }
        else
        { // use the sequential Goertzel algorithm
            double auxs1=0;
            double auxs2=0;
            for(int k=n; k>n1; k--)
            {
                double auxs=b[k]+auxs1*c-auxs2;
                auxs2=auxs1;
                auxs1=auxs;
            }
            s1=_mm512_set_pd(auxs1,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
            s2=_mm512_set_pd(auxs2,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
        }

        // vectorized part of the algorithm
        tc=_mm512_set1_pd(2.0*cosin);

        // "the divide" part of the divid-and-conquer algorithm
        for(int k=m-8; k>=0; k=k-8)
        {

            // load eight column of the 8x8 block
            col0 = _mm512_load_pd(&b[0*m+k+64]);
            col1 = _mm512_load_pd(&b[1*m+k+64]);
            col2 = _mm512_load_pd(&b[2*m+k+64]);
            col3 = _mm512_load_pd(&b[3*m+k+64]);
            col4 = _mm512_load_pd(&b[4*m+k+64]);
            col5 = _mm512_load_pd(&b[5*m+k+64]);
            col6 = _mm512_load_pd(&b[6*m+k+64]);
            col7 = _mm512_load_pd(&b[7*m+k+64]);

            // transpose the 8x8 matrix stored columnwise
            vec8x8_transpose(&col0,&col1,&col2,&col3,&col4,&col5,&col6,&col7);

            // use Goertzel algorithm to process columns of 8x8 matrix of coefficients
            // stored rowwise
            sk=_mm512_fmadd_pd(tc,s1,col7);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col6);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col5);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col4);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col3);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col2);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col1);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col0);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;
        }

        // store two last rows
        _mm512_store_pd(&z[0][0],s2);
        _mm512_store_pd(&z[1][0],s1);

        // compute entries of the 2x2 matrix M
        M[0][0]=-sin((m-1)*x)/sin(x);
        M[1][0]=-sin(m*x)/sin(x);
        M[0][1]=c*sin((m-1)*x)/sin(x)-sin((m-2)*x)/sin(x);
        M[1][1]=c*sin(m*x)/sin(x)-sin((m-1)*x)/sin(x);

        // perform seven matrix-vector multiplications
        vx[0]=z[0][7];
        vx[1]=z[1][7];
        for(int i=6; i>=0; i--)
        {
            tmp[0]=z[0][i]+M[0][0]*vx[0]+M[0][1]*vx[1];
            tmp[1]=z[1][i]+M[1][0]*vx[0]+M[1][1]*vx[1];
            vx[0]=tmp[0];
            vx[1]=tmp[1];
        }

    }
    else
    {
        vx[0]=0;
        vx[1]=0;
    }

    // process first 63  (or less) coefficients
    for(int k=fmin(63,n); k>=1; k--)
    {
        double auxs=b[k]+vx[1]*c-vx[0];
        vx[0]=vx[1];
        vx[1]=auxs;
    }

    // find the sums
    *cx=b[0]+vx[1]*cosin-vx[0];
    *sx=vx[1]*sin(x);

}


/**
 * @brief Implementation of parallel vectorized Goertzel algorithm for x!=k*Pi
 */
void trg_d_p_goertzel(double x,int n,double *b, double *cx, double *sx, int p)
{

    __m512d sk,s1,s2,tc;
    __m512d col0,col1,col2,col3,col4,col5,col6,col7;

    double z[2][8];
    double M[2][2];
    double vx[2];
    double tmp[2];

    double cosin=cos(x);
    double c=2.0*cosin;

    if(n>63) // a part of coefficients will be processed using AVX512
    {

        int n0=p*64*((n-63)/(p*64));
        int n1=n0+63;
        int m=n0/(8*p);
        int r=n-n0;

        double auxs1=0;
        double auxs2=0;

        double *tmpz = (double *)_mm_malloc(2*p*8*64,64);

        if(r==0) // no sequential computation before the vectorized part
        {
            s1=s2=_mm512_setzero_pd();
        }
        else
        { // use the sequential Goertzel algorithm
            for(int k=n; k>n1; k--)
            {
                double auxs=b[k]+auxs1*c-auxs2;
                auxs2=auxs1;
                auxs1=auxs;
            }
        }

        #pragma omp parallel for private(sk,s1,s2,tc,col0,col1,col2,col3,col4,col5,col6,col7) schedule(static) num_threads(p)
        for(int i=0;i<p;i++)
        {

        // vectorized part of the algorithm
        tc=_mm512_set1_pd(2.0*cosin);
        
           if(i==p-1){
              s1=_mm512_set_pd(auxs1,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
              s2=_mm512_set_pd(auxs2,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
           }else{
              s1=_mm512_setzero_pd();
              s2=_mm512_setzero_pd();
           }        
        

        // "the divide" part of the divid-and-conquer algorithm
        for(int k=m-8; k>=0; k=k-8)
        {
            // load 8x8 block
                col0 = _mm512_load_pd(&b[(i*8+0)*m+k+64]);
                col1 = _mm512_load_pd(&b[(i*8+1)*m+k+64]);
                col2 = _mm512_load_pd(&b[(i*8+2)*m+k+64]);
                col3 = _mm512_load_pd(&b[(i*8+3)*m+k+64]);
                col4 = _mm512_load_pd(&b[(i*8+4)*m+k+64]);
                col5 = _mm512_load_pd(&b[(i*8+5)*m+k+64]);
                col6 = _mm512_load_pd(&b[(i*8+6)*m+k+64]);
                col7 = _mm512_load_pd(&b[(i*8+7)*m+k+64]);
            // transpose the 8x8 matrix stored columnwise
            vec8x8_transpose(&col0,&col1,&col2,&col3,&col4,&col5,&col6,&col7);

            // use Goertzel algorithm to process columns of 8x8 matrix of coefficients
            // stored rowwise
            sk=_mm512_fmadd_pd(tc,s1,col7);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col6);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col5);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col4);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col3);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col2);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col1);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;

            sk=_mm512_fmadd_pd(tc,s1,col0);
            sk=_mm512_sub_pd(sk,s2);
            s2=s1;
            s1=sk;
        }

        // store two last rows

         _mm512_store_pd(&tmpz[8*i],s2);
         _mm512_store_pd(&tmpz[8*i+8*p],s1);

        } // end of parallel for

        // compute entries of the 2x2 matrix M
        M[0][0]=-sin((m-1)*x)/sin(x);
        M[1][0]=-sin(m*x)/sin(x);
        M[0][1]=c*sin((m-1)*x)/sin(x)-sin((m-2)*x)/sin(x);
        M[1][1]=c*sin(m*x)/sin(x)-sin((m-1)*x)/sin(x);

        // perform seven matrix-vector multiplications

        vx[0]=tmpz[8*p-1];
        vx[1]=tmpz[8*p+8*p-1];
        for(int i=8*p-2; i>=0; i--)
        {
            tmp[0]=tmpz[i]+M[0][0]*vx[0]+M[0][1]*vx[1];
            tmp[1]=tmpz[8*p+i]+M[1][0]*vx[0]+M[1][1]*vx[1];
            vx[0]=tmp[0];
            vx[1]=tmp[1];
        }

    }
    else
    {
        vx[0]=0;
        vx[1]=0;
    }

    // process first 63  (or less) coefficients
    for(int k=fmin(63,n); k>=1; k--)
    {
        double auxs=b[k]+vx[1]*c-vx[0];
        vx[0]=vx[1];
        vx[1]=auxs;
    }

    // find the sums
    *cx=b[0]+vx[1]*cosin-vx[0];
    *sx=vx[1]*sin(x);

}




/**
 * @brief Implementation of basic sequential Reinsch algorithm.
 */
void trg_d_s_reinsch(double x,int n, double *b, double *cx, double *sx)
{

    double s1,s2,d,d1,u,tmp;
    s2=d1=0;

    if(cos(x)>0)
    {
        tmp=sin(x/2);
        u=-4.0*tmp*tmp;
        for(int k=n; k>=0; k--)
        {
            s1=d1+s2;
            d=b[k]+u*s1+d1;
            d1=d;
            s2=s1;
        }
    }
    else // cos(x)<=0
    {
        tmp=cos(x/2);
        u=4.0*tmp*tmp;
        for(int k=n; k>=0; k--)
        {
            s1=d1-s2;
            d=b[k]+u*s1-d1;
            d1=d;
            s2=s1;
        }
    }
    *sx=s1*sin(x);
    *cx=d-s1*u/2;
}

/**
 * @brief Implementation of vectorized Reinsch algorithm for x!=k*Pi
 */
void trg_d_v_reinsch(double x,int n, double *b, double *cx, double *sx)
{

    double z[2][8];
    double M[2][2];
    double vx[2];
    double vtmp[2];

    __m512d col0,col1,col2,col3,col4,col5,col6,col7;
    __m512d vs2,vs1,vd,vd1,vu;
    double s1,s2,d,d1,u,tmp;
    s2=d1=0;

    int n0=64*((n+1)/64); // coefficients b_{n0-1},...,b_0 will be processed in the vectorized part
    int n1;


    if(cos(x)>0)
    { // case #1
        tmp=sin(x/2);
        u=-4.0*tmp*tmp;

        if(n>=63) // n is large enough - we will use vectorization
        {
            // use the sequential Reinsch algorithm to process "the tail"
            for(int k=n; k>=n0; k--)
            {
                s1=d1+s2;
                d=b[k]+u*s1+d1;
                d1=d;
                s2=s1;
            }

            n1=n0-1;

            // vectorized part

            int m=n0/8;

            vs2=_mm512_set_pd(s2,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
            vd1=_mm512_set_pd(d1,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
            vu =_mm512_set1_pd(u);

            int hh=0;
            // "the divide" part of the divid-and-conquer algorithm

            for(int k=m-8; k>=0; k=k-8)
            {

                // load 8x8 block
                col0 = _mm512_load_pd(&b[0*m+k]);
                col1 = _mm512_load_pd(&b[1*m+k]);
                col2 = _mm512_load_pd(&b[2*m+k]);
                col3 = _mm512_load_pd(&b[3*m+k]);
                col4 = _mm512_load_pd(&b[4*m+k]);
                col5 = _mm512_load_pd(&b[5*m+k]);
                col6 = _mm512_load_pd(&b[6*m+k]);
                col7 = _mm512_load_pd(&b[7*m+k]);

                // transpose the 8x8 matrix stored columnwise
                vec8x8_transpose(&col0,&col1,&col2,&col3,&col4,&col5,&col6,&col7);

                // use Reinsch algorithm to process columns of 8x8 matrix of coefficients
                // stored rowwise
                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col7,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col6,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col5,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col4,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col3,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col2,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col1,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col0,vd);
                vd1=vd;
                vs2=vs1;

            }

            // "the conquer" part of the divid-and-conquer algorithm
            // store two last rows
            _mm512_store_pd(&z[0][0],vs2);
            _mm512_store_pd(&z[1][0],vd1);

            // compute entries of the 2x2 matrix M
            M[0][0]=(-sin(m*x)+sin((m-1)*x))/sin(x);
            M[1][0]=-cos(m*x)+cos((m-1)*x)+u*M[0][0]/2;
            M[0][1]=-sin(m*x)/sin(x);
            M[1][1]=-cos(m*x)+u*M[0][1]/2;

            // perform seven matrix-vector multiplications
            vx[0]=z[0][7];
            vx[1]=z[1][7];
            for(int i=6; i>=0; i--)
            {
                vtmp[0]=z[0][i]-M[0][0]*vx[0]-M[0][1]*vx[1];
                vtmp[1]=z[1][i]-M[1][0]*vx[0]-M[1][1]*vx[1];
                vx[0]=vtmp[0];
                vx[1]=vtmp[1];
            }

            s1=vx[0];
            d= vx[1];


        }
        else
        {   // use sequential Reinsch
            n1=n;
            for(int k=n1; k>=0; k--)
            {
                s1=d1+s2;
                d=b[k]+u*s1+d1;
                d1=d;
                s2=s1;
            }
        }
    }
    else // cos(x)<=0
    {
        tmp=cos(x/2);
        u=4.0*tmp*tmp;

        if(n>=63)
        {
            // use the sequential Reinsch algorithm to process "the tail"
            for(int k=n; k>=n0; k--)
            {
                s1=d1-s2;
                d=b[k]+u*s1-d1;
                d1=d;
                s2=s1;
            }

            n1=n0-1;

            // vectorized part


            int m=n0/8;

            vs2=_mm512_set_pd(s2,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
            vd1=_mm512_set_pd(d1,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
            vu =_mm512_set1_pd(u);

            // "the divide" part of the divid-and-conquer algorithm

            for(int k=m-8; k>=0; k=k-8)
            {


                // load eight column of the 8x8 block
                col0 = _mm512_load_pd(&b[0*m+k]);
                col1 = _mm512_load_pd(&b[1*m+k]);
                col2 = _mm512_load_pd(&b[2*m+k]);
                col3 = _mm512_load_pd(&b[3*m+k]);
                col4 = _mm512_load_pd(&b[4*m+k]);
                col5 = _mm512_load_pd(&b[5*m+k]);
                col6 = _mm512_load_pd(&b[6*m+k]);
                col7 = _mm512_load_pd(&b[7*m+k]);


                // transpose the 8x8 matrix stored columnwise
                vec8x8_transpose(&col0,&col1,&col2,&col3,&col4,&col5,&col6,&col7);

                // use Reinsch algorithm to process columns of 8x8 matrix of coefficients
                // stored rowwise
                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col7);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col6);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col5);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col4);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col3);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col2);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col1);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col0);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

            }

            // "the conquer" part of the divid-and-conquer algorithm
            // store two last rows
            _mm512_store_pd(&z[0][0],vs2);
            _mm512_store_pd(&z[1][0],vd1);

            // compute entries of the 2x2 matrix M
            M[0][0]=(sin(m*x)+sin((m-1)*x))/sin(x);
            M[1][0]=cos(m*x)+cos((m-1)*x)+u*M[0][0]/2;
            M[0][1]=-sin(m*x)/sin(x);
            M[1][1]=-cos(m*x)+u*M[0][1]/2;

            // perform seven matrix-vector multiplications
            vx[0]=z[0][7];
            vx[1]=z[1][7];
            for(int i=6; i>=0; i--)
            {
                vtmp[0]=z[0][i]-M[0][0]*vx[0]-M[0][1]*vx[1];
                vtmp[1]=z[1][i]-M[1][0]*vx[0]-M[1][1]*vx[1];
                vx[0]=vtmp[0];
                vx[1]=vtmp[1];
            }

            s1=vx[0];
            d= vx[1];

        }
        else
        {   // use sequential Reinsch
            n1=n;
            for(int k=n1; k>=0; k--)
            {
                s1=d1-s2;
                d=b[k]+u*s1-d1;
                d1=d;
                s2=s1;
            }
        }

    }

    // find C(x) and S(x)
    *sx=s1*sin(x);
    *cx=d-s1*u/2;
}


/**
 * @brief Implementation of parallel vectorized Reinsch algorithm for x!=k*Pi
 */
void trg_d_p_reinsch(double x,int n, double *b, double *cx, double *sx, int p)
{

    double z[2][8];
    double M[2][2];
    double vx[2];
    double vtmp[2];

    __m512d col0,col1,col2,col3,col4,col5,col6,col7;
    __m512d vs2,vs1,vd,vd1,vu;
    double s1,s2,d,d1,u,tmp;
    s2=d1=0;

    int n0=p*64*((n+1)/(p*64)); // coefficients b_{n0-1},...,b_0 will be processed in the vectorized part
    int n1;

#ifdef _XDEBUG
    printf(">>> n0= %d\n",n0);
#endif

    
    // now we have qxp grid of 8x8 blocks


    double *tmpz = (double *)_mm_malloc(2*p*8*64,64);

    if(cos(x)>0)
    { // case #1
        tmp=sin(x/2);
        u=-4.0*tmp*tmp;

        if(n>=63) // n is large enough - we will use vectorization
        {
            // use the sequential Reinsch algorithm to process "the tail"
            for(int k=n; k>=n0; k--)
            {
                s1=d1+s2;
                d=b[k]+u*s1+d1;
                d1=d;
                s2=s1;
            }

            n1=n0-1;

            // vectorized part

            int m=n0/(8*p);


#ifdef _XDEBUG
           printf(">>> m = %d\n",m);
           printf(">>> q = %d\n",m/8);
#endif


            // parallel loop
            
            #pragma omp parallel for private(vs2,vs1,vd,vd1,vu,col0,col1,col2,col3,col4,col5,col6,col7) schedule(static) num_threads(p)
            for(int i=0;i<p;i++){


            if(i==p-1){
              vs2=_mm512_set_pd(s2,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
              vd1=_mm512_set_pd(d1,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
            }else{
              vs2=_mm512_setzero_pd();
              vd1=_mm512_setzero_pd();            
            }
            
            
            vu =_mm512_set1_pd(u);

            // "the divide" part of the divid-and-conquer algorithm

            for(int k=m-8; k>=0; k=k-8)
            {


                // load eight column of the 8x8 block
                col0 = _mm512_load_pd(&b[(i*8+0)*m+k]);
                col1 = _mm512_load_pd(&b[(i*8+1)*m+k]);
                col2 = _mm512_load_pd(&b[(i*8+2)*m+k]);
                col3 = _mm512_load_pd(&b[(i*8+3)*m+k]);
                col4 = _mm512_load_pd(&b[(i*8+4)*m+k]);
                col5 = _mm512_load_pd(&b[(i*8+5)*m+k]);
                col6 = _mm512_load_pd(&b[(i*8+6)*m+k]);
                col7 = _mm512_load_pd(&b[(i*8+7)*m+k]);


                // transpose the 8x8 matrix stored columnwise
                vec8x8_transpose(&col0,&col1,&col2,&col3,&col4,&col5,&col6,&col7);

                // use Reinsch algorithm to process columns of 8x8 matrix of coefficients
                // stored rowwise
                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col7,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col6,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col5,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col4,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col3,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col2,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col1,vd);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_add_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,vd1);
                vd =_mm512_add_pd(col0,vd);
                vd1=vd;
                vs2=vs1;

            }

            // "the conquer" part of the divid-and-conquer algorithm
            // store two last rows
            _mm512_store_pd(&tmpz[8*i],vs2);
            _mm512_store_pd(&tmpz[8*i+8*p],vd1);


            }

            // compute entries of the 2x2 matrix M
            M[0][0]=(-sin(m*x)+sin((m-1)*x))/sin(x);
            M[1][0]=-cos(m*x)+cos((m-1)*x)+u*M[0][0]/2;
            M[0][1]=-sin(m*x)/sin(x);
            M[1][1]=-cos(m*x)+u*M[0][1]/2;

            // perform seven matrix-vector multiplications
            vx[0]=tmpz[8*p-1];
            vx[1]=tmpz[8*p+8*p-1];
            for(int i=8*p-2; i>=0; i--)
            {
                vtmp[0]=tmpz[i]-M[0][0]*vx[0]-M[0][1]*vx[1];
                vtmp[1]=tmpz[8*p+i]-M[1][0]*vx[0]-M[1][1]*vx[1];
                vx[0]=vtmp[0];
                vx[1]=vtmp[1];
            }

            s1=vx[0];
            d= vx[1];


        }
        else
        {   // use sequential Reinsch
            n1=n;
            for(int k=n1; k>=0; k--)
            {
                s1=d1+s2;
                d=b[k]+u*s1+d1;
                d1=d;
                s2=s1;
            }
        }
    }
    else // cos(x)<=0
    {
        tmp=cos(x/2);
        u=4.0*tmp*tmp;

        if(n>=63)
        {
            // use the sequential Reinsch algorithm to process "the tail"
            for(int k=n; k>=n0; k--)
            {
                s1=d1-s2;
                d=b[k]+u*s1-d1;
                d1=d;
                s2=s1;
            }

            n1=n0-1;

            // vectorized part


            int m=n0/(8*p);

            // parallel loop

            #pragma omp parallel for private(vs2,vs1,vd,vd1,vu,col0,col1,col2,col3,col4,col5,col6,col7) schedule(static) num_threads(p)
            for(int i=0;i<p;i++){
	    

            if(i==p-1){
              vs2=_mm512_set_pd(s2,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
              vd1=_mm512_set_pd(d1,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
            }else{
              vs2=_mm512_setzero_pd();
              vd1=_mm512_setzero_pd();
            }

	    
            vu =_mm512_set1_pd(u);

            // "the divide" part of the divid-and-conquer algorithm

            for(int k=m-8; k>=0; k=k-8)
            {
                // load eight column of the 8x8 block

                col0 = _mm512_load_pd(&b[(i*8+0)*m+k]);
                col1 = _mm512_load_pd(&b[(i*8+1)*m+k]);
                col2 = _mm512_load_pd(&b[(i*8+2)*m+k]);
                col3 = _mm512_load_pd(&b[(i*8+3)*m+k]);
                col4 = _mm512_load_pd(&b[(i*8+4)*m+k]);
                col5 = _mm512_load_pd(&b[(i*8+5)*m+k]);
                col6 = _mm512_load_pd(&b[(i*8+6)*m+k]);
                col7 = _mm512_load_pd(&b[(i*8+7)*m+k]);



                // transpose the 8x8 matrix stored columnwise
                vec8x8_transpose(&col0,&col1,&col2,&col3,&col4,&col5,&col6,&col7);

                // use Reinsch algorithm to process columns of 8x8 matrix of coefficients
                // stored rowwise
                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col7);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col6);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col5);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col4);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col3);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col2);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col1);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

                vs1=_mm512_sub_pd(vd1,vs2);
                vd =_mm512_fmadd_pd(vu,vs1,col0);
                vd =_mm512_sub_pd(vd,vd1);
                vd1=vd;
                vs2=vs1;

            }


            // "the conquer" part of the divid-and-conquer algorithm
            // store two last rows
            _mm512_store_pd(&tmpz[8*i],vs2);
            _mm512_store_pd(&tmpz[8*i+8*p],vd1);


            }
	    

            // compute entries of the 2x2 matrix M
            M[0][0]=(sin(m*x)+sin((m-1)*x))/sin(x);
            M[1][0]=cos(m*x)+cos((m-1)*x)+u*M[0][0]/2;
            M[0][1]=-sin(m*x)/sin(x);
            M[1][1]=-cos(m*x)+u*M[0][1]/2;

            // perform seven matrix-vector multiplications

            vx[0]=tmpz[8*p-1];
            vx[1]=tmpz[8*p+8*p-1];
            for(int i=8*p-2; i>=0; i--)
            {
                vtmp[0]=tmpz[i]-M[0][0]*vx[0]-M[0][1]*vx[1];
                vtmp[1]=tmpz[8*p+i]-M[1][0]*vx[0]-M[1][1]*vx[1];
                vx[0]=vtmp[0];
                vx[1]=vtmp[1];
            }
	    

            s1=vx[0];
            d= vx[1];

        }
        else
        {   // use sequential Reinsch
            n1=n;
            for(int k=n1; k>=0; k--)
            {
                s1=d1-s2;
                d=b[k]+u*s1-d1;
                d1=d;
                s2=s1;
            }
        }

    }

    // find C(x) and S(x)
    *sx=s1*sin(x);
    *cx=d-s1*u/2;
}


