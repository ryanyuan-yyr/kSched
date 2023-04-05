struct MatrixTransposeArgs {
    float* idata, *odata;
    int width;
};

struct MatrixTransposeContext{
    float* h_idata;
    float* h_odata;
};
