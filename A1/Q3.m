syms theta l r real

G = [cos(theta + pi/2)    sin(theta + pi/2)    0.25;
     cos(theta + 7*pi/6)  sin(theta + 7*pi/6)  0.25;
     cos(theta + 11*pi/6) sin(theta + 11*pi/6) 0.25];

G = 1/0.1 * G;

q_dot = [cos(theta) sin(theta) 1]';

u = G * q_dot;

u = simplify(u)

pretty(u)


