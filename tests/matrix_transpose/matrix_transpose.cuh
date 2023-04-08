struct MatrixTransposeArgs {
    float* idata, *odata;
    unsigned long width;
};

struct MatrixTransposeContext{
    float* h_idata;
    float* h_odata;
};
