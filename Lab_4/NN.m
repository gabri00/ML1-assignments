% Configurations
clear
clc
addpath('mnist\');
addpath('plot\');

% Subset percentage
p = 0.05;
% Number of units in the hidden layer
nh = 2;
% Classes to confront
classes = [
    1 8;
    3 9;
    1 7;
    0 6;
    2 5
];

for i = 1:length(classes)
    % Load data
    [X, T] = loadMNIST(0, classes(i, :));

    % Srink dataset size, take only p% of the length
    % Then shuffle the data
    subset_size = floor(length(X)*p);
    idx = randperm(length(X), subset_size);
    X = X(idx, :);
    T = T(idx, :);

    % Train autoencoder
    autoEncoder = trainAutoencoder(X', nh);
    encoded_data = encode(autoEncoder, X');

    % Plot data
    figure
    plotcl(encoded_data', [T(T==classes(i, 1), :); T(T==classes(i, 2), :)]);

    legend(['Digit ', num2str(classes(i, 1))], ['Digit ', num2str(classes(i, 2))]);
    xlabel('Hidden unit 1');
    ylabel('Hidden unit 2');
    title(['Autoencoder on digits ', num2str(classes(i, 1)), ' and ', num2str(classes(i, 2))]);
end