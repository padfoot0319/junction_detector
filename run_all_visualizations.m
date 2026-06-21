function run_all_visualizations()
% Visualize all CPU and GPU outputs under exports/.

thisFile = mfilename('fullpath');
rootDir = fileparts(thisFile);

outDir = fullfile(rootDir, 'visualizations');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

files = [
    dir(fullfile(rootDir, 'exports', 'cpu_config*.json'));
    dir(fullfile(rootDir, 'exports', 'gpu_config*.json'))
];

failed = strings(0);

for i = 1:numel(files)
    jsonPath = fullfile(files(i).folder, files(i).name);

    try
        run_road_network(jsonPath);
        close all force;
    catch ME
        failed(end+1) = string(files(i).name); %#ok<AGROW>
        fprintf(2, "Failed to visualize %s: %s\n", files(i).name, ME.message);
        close all force;
    end
end

if ~isempty(failed)
    error("Some visualizations failed: %s", strjoin(failed, ", "));
end

fprintf("Created %d visualizations.\n", numel(files));
end