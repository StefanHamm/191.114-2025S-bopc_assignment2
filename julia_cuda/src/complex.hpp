#ifndef COMPLEX
#define COMPLEX

class Complex {
    public:
        float real;
        float imag;

        Complex(float real, float imag) {
            this->real = real;
            this->imag = imag;
        }

        float magnitude2(void) {
            return (this->real * this->real) + (this->imag * this->imag);
        }
        Complex operator* (const Complex num) {
            return Complex((this->real * num.real) - (this->imag * num.imag), (this->imag * num.real) + (this->real * num.imag));
        }

        Complex operator+ (const Complex num) {
            return Complex(this->real + num.real, this->imag + num.imag);
        }
};

#endif