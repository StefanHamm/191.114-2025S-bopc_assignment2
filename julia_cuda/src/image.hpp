#ifndef IMAGE
#define IMAGE

class Pixel {
    public:
        char r;
        char g;
        char b;

        Pixel() {}
        Pixel(char r, char g, char b) {
            this -> r = r;
            this -> g = g;
            this -> b = b;
        }

        void assignValueToPixel(char, char, char);
};

class Image {
    public:
        int w, h;
        Pixel *pixels;

        Image() {}
        Image(int w, int h) {
            this -> w = w;
            this -> h = h;
        }

        bool createPixels(void);
        int getImageSize(void);
        void saveImage(const char *);
        Pixel *getPixelArrayPointer(void);

        ~Image();
};

#endif