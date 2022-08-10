% Basic Operations
a = pi
disp(sprintf('2 decimals: %0.2f', a))
disp(sprintf('2 decimals: %0.6f', a))
format long
a
format short
a
A = [1 2; 3 4; 5 6]
v = [1 2 3]
v = 1:0.1:2
ones(2,3)
c = 2*ones(2,3)
w = ones(1,3)
w = zeros(1,3)
w = rand(1,3)
rand(3,3)
w = rand(1,3)
ww = -6 + sqrt(10)*(randn(1,10000));
hist(ww)
hist(ww, 50)
eye(4)


% Move Data Around
size(A)
size(A,1)
size(A,2)
who
whos
%{
   Loading files in Matlab:
    cd to the folder, then use "load filename" to load
%}
A(2,:) % get the 2nd row
A([1 3], :) %get the first and the third row
A(:, 2) % get the 2nd column
A = [A, [100; 101; 102]]; % append another column vector to right
A(:) % put all elements of A into a single vector
A = [1 2; 3 4; 5 6]
B = [11 12; 13 14; 15 16]
C = [A ; B]
D = [A B]



% Plotting Data
t = [0:0.01:0.98];
y1 = sin(2*pi*4*t);
plot(t, y1)
hold on;
y2 = cos(2*pi*4*t);
plot(t, y2, 'r')
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('my plot')
print -dpng 'myPlot.png'
close

figure(1); plot(t, y1);
figure(2); plot(t, y2);

subplot(1,2,1); % Divides plot a 1x2 grid, access first element
plot(t, y1);
subplot(1,2,2);
plot(t, y2);
axis([0.5 1 -1 1])
close;

K = magic(5)
imagesc(K);
imagesc(K), colorbar, colormap gray;
close;

% Control statements and Function
v = zeros(10, 1)
for i=1:10
    v(i) = 2^i;
end
v

i = 1;
while i <= 5
    v(i) = 100;
    i = i + 1;
end
v

i = 1;
while true
    v(i) = 999;
    i = i + 1;
    if i == 6
        break;
    end
end
v

if v(1) == 1
    disp('the value is one')
elseif v(1) == 999
    disp('the value is 999')
else
    disp('other values')
end

[a,b] = squareNum(5)
a
b


X = [1 1; 1 2; 1 3]
y = [1; 2; 3]
theta = [0;0];
J = costFunctionJ(X, y, theta)



% Vectorization

