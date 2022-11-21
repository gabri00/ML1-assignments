clear
clc

addpath('mnist\');
addpath('plot\');

p = 0.05;
classes = [
    1, 8;
    3, 8;
    1, 7;
    5, 6
];

for i = 1:size(classes, 1)
    [X_a, T_a] = loadMNIST(0, classes(i, 1));
    [X_b, T_b] = loadMNIST(0, classes(i, 2));

    % Shuffle sets
    subset_size = floor(size(X_a, 1)*p);
    idx_a = randperm(size(X_a, 1), subset_size);
    idx_b = randperm(size(X_b, 1), subset_size);

    dataset = [
        X_a(idx_a, :);
        X_b(idx_b, :)
    ];

    % Train autoencoder
    autoEncoder = trainAutoencoder(dataset', 2);
    encoded_data = encode(autoEncoder, dataset');

    % Plot data
    figure
    plotcl(encoded_data', [T_a(1:subset_size); T_b(1:subset_size)]);
    legend(['Class ', num2str(T_a(1))], ['Class ', num2str(T_b(1))]);
    xlabel('Hidden unit 1');
    ylabel('Hidden unit 2');
    title(['Autoencoder classes ', num2str(classes(i, 1)), ' and ', num2str(classes(i, 2))]);
end