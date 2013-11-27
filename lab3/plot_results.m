% input_size input_set input_try variant nb_threads try global_start_sec global_start_nsec global_stop_sec global_stop_nsec
addpath matlab
results = select(data, [1]);
results = groupby(results, [1]);
results = reduce(results, {@none});
results = duplicate(results, [2]);

% Get Global mean runtime, Global standard deviation, mean standard deviation accross input instances, standard deviation of means accross patterns from results from groups
res_size = size(results);
for i = 1:res_size(1)
	line = results(i, :);
	group = line(1);
	mat = where(data, [1], {[group]});
	line = [line(:, 1) points(mat)];
	results(i, :) = line;
end

prepad = [zeros(res_size(1), 1) zeros(size(results))];
postpad = [ones(res_size(1), 1) .* 2 zeros(size(results))];
results = [ones(res_size(1), 1) results]

ref = max(results(:, 3));
results(:, 3) = results(:, 3) ./ ref;
results = [prepad; results; postpad];

quickbar(1, results, 1, 3, 0, 'grouped', 0.5, 'MgOpenModernaBold.ttf', 25, 800, 400, 'Group', 'Score', 'Score per group for parallel sorting algorithms',group_names(), 'northeast', 'final_scores.eps', 'epsc');
