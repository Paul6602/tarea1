#ifndef TENSORTRANSFORM_H
#define TENSORTRANSFORM_H

class TensorTransform {
public:

    virtual double apply(double x) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    double apply(double x) const override {
        return (x > 0) ? x : 0.0;
    }
};

class Sigmoid : public TensorTransform {
public:
    double apply(double x) const override {
        return 1.0 / (1.0 + exp(-x));
    }
};

#endif
