
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <time.h>
#include <fstream>

#include "image.hpp"
#include "tinycolormap.hpp"
#include "cli_parser.hpp"
#include "kernel.hpp"

#include <chrono>
using namespace std::chrono;

//#define CMAP tinycolormap::ColormapType::Gray
#define CMAP tinycolormap::ColormapType::Magma

int global_block_x = -1;
int global_block_y = -1;

void write_image(float *julia_set, const int res_x, const int res_y, std::string filename) {

    // create image
    Image im(res_x, res_y);
    if(im.createPixels() == false) {
        std::cerr << "Could not allocate pixel array";
        exit(-1);
    }

    Pixel *pixels = im.getPixelArrayPointer();

    // set value for pixels
    for(int x = 0; x < res_x; x++) {
        for(int y = 0; y < res_y; y++) {    

            int arr_offset = x + ((res_y - y - 1) * res_x);

            const tinycolormap::Color color = tinycolormap::GetColor(julia_set[x*res_y+y], CMAP);
            pixels[arr_offset].r = color.ri();
            pixels[arr_offset].g = color.gi();
            pixels[arr_offset].b = color.bi();
        }
    }


    std::cout << "Saving Image..." << std::endl; 
    im.saveImage(filename.c_str());    
}

// argument parsing utility functions

void usage(const char* argv[]) {
    std::cout << "Usage: " << argv[0] << " [ARGS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Optional Arguments (ARGS):" << std::endl;
    std::cout << "\t -f, --filename FILENAME: \t\t file name for output image (PPM Format)" << std::endl;
    std::cout << "\t -c, --constant C_REAL C_IMAG: \t\t real and imaginary component for constant C" << std::endl;
    std::cout << "\t -r, --resolution RES_X RES_Y: \t\t horizontal and vertical image resolution" << std::endl;
    std::cout << "\t -s, --scale SCALE: \t\t\t image scale" << std::endl;
    std::cout << "\t -i, --max-iterations ITERATIONS: \t maximum number of iterations" << std::endl;
    std::cout << "\t -m, --max-magnitude MAGNITUDE: \t magnitude cutoff" << std::endl;
    std::cout << "\t -b, --blocksize BLOCK_X BLOCK_Y \t block size for kernel launch" << std::endl;
    std::cout << "\t -n, --nrep NREPS \t number of repetitions" << std::endl;
}


int main(int argc, char const *argv[]) {

    // default argument values
    float c_real = -0.4;
    float c_imag = 0.6;
    //float scale = 1.5;
    float scale = 0.5;
    int res_x = 1000;
    int res_y = 1000;  
    //int max_iter = 500;
    int max_iter = 200;
    float max_mag = 10;
    std::string filename = "julia.ppm";
    std::string csvname = "";
    bool save_file = false;
    bool csv_file = false;
    int nreps = 1;

    // parse arguments
    try {

        for (int i=1; i<argc-1; i++) {
            if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--filename")) {
                if (i+1 < argc) {
                    filename = argv[++i];
                    save_file = true;
                } else {
                    //usage(argv);
                    throw std::invalid_argument("Failed to parse filename");
                }
            } else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "--constant")) {
                i++;
                float* cst = parseFloats(argc, argv, &i, 2, "constant");
                c_real = cst[0];
                c_imag = cst[1];
                free(cst);
            } else if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--resolution")) {
                i++;
                int* r = parseInts(argc, argv, &i, 2, "resolution");
                res_x = r[0];
                res_y = r[1];
                free(r);
            } else if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--scale")) {
                i++;
                float* s = parseFloats(argc, argv, &i, 1, "scale");
                scale = s[0];
                free(s);
            } else if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--max-iterations")) {
                i++;
                int* iter = parseInts(argc, argv, &i, 1, "max-iterations");
                max_iter = iter[0];
                free(iter);
            } else if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--max-magnitude")) {
                i++;
                float* mag = parseFloats(argc, argv, &i, 1, "max-magnitude");
                max_mag = mag[0];
                free(mag);
            } else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--blocksize")) {
                i++;
                int* r = parseInts(argc, argv, &i, 2, "blocksize");
                global_block_x = r[0];
                global_block_y = r[1];
                free(r);
            } else if (!strcmp(argv[i], "-n") || !strcmp(argv[i], "--nrep")) {
                i++;
                int* nrep_ar = parseInts(argc, argv, &i, 1, "nrep");
                nreps = nrep_ar[0];
                free(nrep_ar);
            } else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output")) {
                if (i+1 < argc) {
                    csvname = argv[++i];
                    csv_file = true;
                } else {
                    //usage(argv);
                    throw std::invalid_argument("Failed to parse csv file name");
                }
            } else {
                std::cout << argv[i] << std::endl;
                throw std::invalid_argument("Invalid argument");
            }
        }

    } catch (std::invalid_argument& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl << std::endl;
        usage(argv);
        return -1;
    }

    // ===== IMAGE COMPUTATION ===== //

    Complex c(c_real, c_imag);

    int max_dim = res_x > res_y ? res_x : res_y;
    float x_scale = (float) res_x / max_dim;
    float y_scale = (float) res_y / max_dim;

    float *julia_set = new float[res_x*res_y];
        

    //std::ostream& os = std::cout;

    std::ostringstream oss;

    for(int i=0; i<nreps; i++) {
        auto start = high_resolution_clock::now();
        julia_kernel(julia_set, c, scale, res_x, res_y, max_iter, max_mag, x_scale, y_scale);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        oss  << i
            << ";" 
            << res_x
            << ";" 
            << res_y
            << ";" 
            << scale
            << ";" 
            << global_block_x
            << ";" 
            << global_block_y
            << ";" 
            << duration.count() 
            << std::endl;
    }    

    if(csv_file) {
        std::ofstream ofs;
        ofs.open(csvname);
        ofs << oss.str();
        ofs.close();
    } else {
        std::cout << oss.str();
    }


    if( save_file ) {
        write_image(julia_set, res_x, res_y, filename);
    }    

    delete [] julia_set;


    return 0;
}

