#include "image.hpp"

#include <iostream>
#include <ostream>

void Pixel::assignValueToPixel(char r, char g, char b) {
    this -> r = r;
    this -> g = g;
    this -> b = b;
}

bool Image::createPixels(void) {
    this -> pixels = new Pixel[(this -> w) * (this -> h)];
    if(this -> pixels == NULL) {
        std::cout << "Error creating image" << std::endl;
        return false;
    }
    return true;
}

Pixel *Image::getPixelArrayPointer(void) {
    return this -> pixels;
}

int Image::getImageSize(void) {
    return (this -> w) * (this -> h);
}

void Image::saveImage(const char* filename) {
    FILE *file = fopen(filename, "wb");
    fprintf(file, "P6\n%d %d\n255\n", this->w, this->h);
    for(int i = 0; i < ((this->w * (this->h))); i++) {
        char color[3] = { 0 };
        color[0] = (char) pixels[i].r;
        color[1] = (char) pixels[i].g;
        color[2] = (char) pixels[i].b;
        fwrite(color, 1, 3, file);
    }
    fclose(file);
} 

Image::~Image() {
    //delete[] this -> pixels;
}