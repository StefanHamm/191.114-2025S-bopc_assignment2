#ifndef COMPLEX_GPU
#define COMPLEX_GPU

class Complex {
    public:
        float real;
        float imag;

        __host__ __device__ Complex(float real, float imag) {
            this->real = real;
            this->imag = imag;
        }

        __host__ __device__ float magnitude2(void) {
            return (this->real * this->real) + (this->imag * this->imag);
        }
        __host__ __device__ Complex operator* (const Complex num) {
            return Complex((this->real * num.real) - (this->imag * num.imag), (this->imag * num.real) + (this->real * num.imag));
        }

        __host__ __device__ Complex operator+ (const Complex num) {
            return Complex(this->real + num.real, this->imag + num.imag);
        }
};

#endif