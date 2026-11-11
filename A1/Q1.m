clear; clc;
syms l1 l2 lb alpha gamma theta x y real

%% Part A: Symbolic transformation matrices
G_2E = [ 0  1  l2;
        -1  0  0;
         0  0  1];

G_12 = [cos(gamma) -sin(gamma) l1;
        sin(gamma)  cos(gamma) 0;
        0           0          1];

G_B1 = [cos(alpha) -sin(alpha) lb;
        sin(alpha)  cos(alpha) 0;
        0           0          1];

G_SB = [cos(theta) -sin(theta) x;
        sin(theta)  cos(theta) y;
        0           0          1];

G_SE = simplify(G_SB * G_B1 * G_12 * G_2E);

disp('--- Part A ---');
disp('G_2E ='); pretty(G_2E)
disp('G_12 ='); pretty(G_12)
disp('G_B1 ='); pretty(G_B1)
disp('G_SB ='); pretty(G_SB)
disp('G_SE ='); pretty(G_SE)

%% Part B: Points from {E} to {s}
vars = [theta, alpha, gamma, l1, l2, lb, x, y];
vals = [-pi/2, pi/6, pi/4, 1, 1, 0.5, 0.8, 0.8];

G_SE_num = double(subs(G_SE, vars, vals));

% Homogeneous coordinates
p1_E = [0; 0; 1];
p2_E = [0.5; 1; 1];

p1_S = G_SE_num * p1_E;
p2_S = G_SE_num * p2_E;

disp('--- Part B ---');
fprintf('p1 in {s}: (%.4f, %.4f)\n', p1_S(1), p1_S(2));
fprintf('p2 in {s}: (%.4f, %.4f)\n', p2_S(1), p2_S(2));

%% Part C: Point q^B expressed in {E}
qB = [1; 2; 1];

% Transform from {E} to {B}, then invert
G_BE = G_B1 * G_12 * G_2E;
G_BE_num = double(subs(G_BE, vars, vals));

qE = G_BE_num \ qB;

disp('--- Part C ---');
fprintf('q in {E}: (%.4f, %.4f)\n', qE(1), qE(2));
