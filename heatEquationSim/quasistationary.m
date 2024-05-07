function th = quasistationary(v, x, y, th0)
    alpha = 0.14;
    Q = 100;
    k = 0.19;
    r = sqrt(x^2 + y^2);
    w = v * r / (2 * alpha);
    k0 = besselk(0, w);
    th = th0 + Q * v / (2 * pi * k) * exp(-x * v / (2 * alpha)) * k0;
    
