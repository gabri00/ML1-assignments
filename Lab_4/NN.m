wine_dataset = importdata('wine.data');
wine_input = wine_dataset(:, 2:size(wine_dataset, 2))';
wine_target = wine_dataset(:, 1)';

classes = unique(wine_target);

for i = 1:size(classes)
    wine_target(i) = wine_target()

