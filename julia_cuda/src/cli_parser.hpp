#ifndef CLI_PARSER
#define CLI_PARSER

int* parseInts(int argc, char const *argv[], int* cur_arg, int num_ints, const char* argname);
float* parseFloats(int argc, char const *argv[], int* cur_arg, int num_floats, const char* argname);

#endif