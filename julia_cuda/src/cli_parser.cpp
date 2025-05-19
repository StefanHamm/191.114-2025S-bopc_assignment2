
#include <string>
#include <sstream>
#include <cstdio>
#include "cli_parser.hpp"


float* parseFloats(int argc, char const *argv[], int* cur_arg, int num_floats, const char* argname) {
    float* parsedFloats = (float*) malloc(num_floats*sizeof(float));
    for (int i=0; i<num_floats; i++) {
        if (*cur_arg < argc) {
            try {
                parsedFloats[i] = std::stof(argv[*cur_arg]);
            } catch (std::invalid_argument& e) {
                std::stringstream fmt;
                fmt << "Failed to parse float parameter \"" << argv[*cur_arg] << "\" for argument \"" << argname << std::endl;
                throw std::invalid_argument(fmt.str());
            }
            (*cur_arg)++;
        } else {
            std::stringstream fmt;
            fmt << "Insufficient number of parameters for argument \"" << argname << "\"" << std::endl;
            throw std::invalid_argument(fmt.str());
        }
    }
    // decrease once
    (*cur_arg)--; 
    return parsedFloats;
}

int* parseInts(int argc, char const *argv[], int* cur_arg, int num_ints, const char* argname) {
    int* parsedInts = (int*) malloc(num_ints*sizeof(int));
    for (int i=0; i<num_ints; i++) {
        if (*cur_arg < argc) {
            try {
                parsedInts[i] = std::stoi(argv[*cur_arg]);
            } catch (std::invalid_argument& e) {
                std::stringstream fmt;
                fmt << "Failed to parse integer parameter \"" << argv[*cur_arg] << "\" for argument \"" << argname << "\"" << std::endl;
                throw std::invalid_argument(fmt.str());
            }
            (*cur_arg)++;
        } else {
            std::stringstream fmt;
            fmt << "Insufficient number of parameters for argument \"" << argname << "\"" << std::endl;
            throw std::invalid_argument(fmt.str());
        }
    }
    // decrease once
    (*cur_arg)--; 
    return parsedInts;
}
