function run_road_network(jsonPath)
% Visualization script for automatically detected junction candidates.

set(0, 'DefaultFigureVisible', 'off');

thisFile = mfilename('fullpath');
rootDir  = fileparts(thisFile);

if nargin < 1 || isempty(jsonPath)
    jsonPath = fullfile(rootDir, 'exports', 'gpu_config12.json');
end

if ~isfile(jsonPath)
    jsonPath = fullfile(rootDir, jsonPath);
end

visualizationDir = fullfile(rootDir, 'visualizations');
if ~exist(visualizationDir, 'dir')
    mkdir(visualizationDir);
end

[~, jsonName, ~] = fileparts(jsonPath);
imagePath = fullfile(visualizationDir, strcat(jsonName, '.png'));

showLabels = true;
showLegend = true;
showNodes = false;

backgroundColor = [0.42 0.42 0.42];    % gray roads outside detected junctions
primaryColor    = [0.96 0.96 0.96];    % white: primary/boundary roads in junctions

backgroundLineWidth = 0.8;
primaryLineWidth    = 2.2;
secondaryLineWidth  = 3.0;

fig = plotDetectedJunctions(jsonPath, showLabels, showLegend, showNodes, ...
    backgroundColor, primaryColor, ...
    backgroundLineWidth, primaryLineWidth, secondaryLineWidth);

drawnow;
exportgraphics(fig, imagePath, 'Resolution', 200);
close(fig);
close all force;
end


function fig = plotDetectedJunctions(jsonPath, showLabels, showLegend, showNodes, ...
    backgroundColor, primaryColor, ...
    backgroundLineWidth, primaryLineWidth, secondaryLineWidth)

    if ~isfile(jsonPath)
        error("JSON file not found: %s", jsonPath);
    end

    D = jsondecode(fileread(jsonPath));

    if ~isfield(D, "interchanges") || isempty(D.interchanges)
        error("No 'interchanges' field found in JSON: %s", jsonPath);
    end

    Js = D.interchanges;
    nJ = numel(Js);
    C = lines(max(nJ, 1));

    fig = figure('Visible', 'off', 'Color', [0.08 0.08 0.08]);
    ax = axes('Parent', fig);
    hold(ax, 'on');
    grid(ax, 'on');
    axis(ax, 'equal');
    set(ax, 'Color', [0.05 0.05 0.05]);
    set(ax, 'XColor', [0.85 0.85 0.85], 'YColor', [0.85 0.85 0.85]);
    xlabel(ax, 'x [m]');
    ylabel(ax, 'y [m]');
    title(ax, 'Detected Complex Junctions', ...
        'Color', [0.9 0.9 0.9]);

    allX = [];
    allY = [];

    % 1) Draw all available road links in gray.
    hasBackground = isfield(D, "background_graph") && ~isempty(D.background_graph);

    if hasBackground
        [bgNodeMap, bgNodeXY] = buildNodeMap(D.background_graph);

        if isfield(D.background_graph, "edges") && ~isempty(D.background_graph.edges)
            Ebg = D.background_graph.edges;
            for i = 1:numel(Ebg)
                P = polylineFromGraphEdge(Ebg(i), bgNodeMap);
                if isempty(P)
                    continue;
                end

                plot(ax, P(:,1), P(:,2), '-', ...
                    'Color', backgroundColor, ...
                    'LineWidth', backgroundLineWidth);

                allX = [allX; P(:,1)]; %#ok<AGROW>
                allY = [allY; P(:,2)]; %#ok<AGROW>
            end
        end

        if showNodes && ~isempty(bgNodeXY)
            scatter(ax, bgNodeXY(:,1), bgNodeXY(:,2), 8, backgroundColor, 'filled');
        end
    else
        warning("No background_graph field in JSON. Gray non-junction roads cannot be drawn.");
    end

    % 2) Draw detected junction links on top.
    legendHandles = gobjects(0);
    legendLabels = strings(0);

    hBg = plot(ax, nan, nan, '-', 'Color', backgroundColor, ...
        'LineWidth', backgroundLineWidth);
    legendHandles(end+1) = hBg; %#ok<AGROW>
    legendLabels(end+1) = "non-junction roads"; %#ok<AGROW>

    hPr = plot(ax, nan, nan, '-', 'Color', primaryColor, ...
        'LineWidth', primaryLineWidth);
    legendHandles(end+1) = hPr; %#ok<AGROW>
    legendLabels(end+1) = "primary/boundary roads"; %#ok<AGROW>

    for k = 1:nJ
        J = Js(k);
        jid = getJunctionId(J, k);
        col = C(k, :);

        primaryIds = getNumericIdList(J, "primary_boundary_link_ids");
        secondaryIds = getNumericIdList(J, "secondary_link_ids");

        ptsForLabel = [];
        plottedSecondary = false;

        if isfield(J, "graph") && ~isempty(J.graph)
            [nodeMap, nodeXY] = buildNodeMap(J.graph);

            if isfield(J.graph, "edges") && ~isempty(J.graph.edges)
                E = J.graph.edges;

                for i = 1:numel(E)
                    lid = getEdgeLinkId(E(i));
                    role = getEdgeRole(E(i));

                    isPrimary = false;
                    if role == "main" || role == "primary"
                        isPrimary = true;
                    elseif ~isempty(primaryIds) && ~isnan(lid) && any(primaryIds == lid)
                        isPrimary = true;
                    end

                    isSecondary = false;
                    if role == "ramp" || role == "secondary" || role == "interior"
                        isSecondary = true;
                    elseif ~isempty(secondaryIds) && ~isnan(lid) && any(secondaryIds == lid)
                        isSecondary = true;
                    end

                    if ~isPrimary && ~isSecondary
                        isSecondary = true;
                    end

                    P = polylineFromGraphEdge(E(i), nodeMap);
                    if isempty(P)
                        continue;
                    end

                    if isPrimary
                        plot(ax, P(:,1), P(:,2), '-', ...
                            'Color', primaryColor, ...
                            'LineWidth', primaryLineWidth);
                    else
                        plot(ax, P(:,1), P(:,2), '-', ...
                            'Color', col, ...
                            'LineWidth', secondaryLineWidth);
                        plottedSecondary = true;
                    end

                    ptsForLabel = [ptsForLabel; P]; %#ok<AGROW>
                    allX = [allX; P(:,1)]; %#ok<AGROW>
                    allY = [allY; P(:,2)]; %#ok<AGROW>
                end
            end

            if showNodes && ~isempty(nodeXY)
                scatter(ax, nodeXY(:,1), nodeXY(:,2), 12, col, 'filled');
            end
        else
            warning("J%d has no graph field. It cannot be plotted in graph-edge mode.", jid);
        end

        if plottedSecondary
            h = plot(ax, nan, nan, '-', 'Color', col, ...
                'LineWidth', secondaryLineWidth);
            legendHandles(end+1) = h; %#ok<AGROW>
            legendLabels(end+1) = sprintf("J%d secondary/interior", jid); %#ok<AGROW>
        end

        if showLabels && ~isempty(ptsForLabel)
            cx = mean(ptsForLabel(:,1), 'omitnan');
            cy = mean(ptsForLabel(:,2), 'omitnan');
            text(ax, cx, cy, sprintf('J%d', jid), ...
                'Color', col, ...
                'FontWeight', 'bold', ...
                'FontSize', 12, ...
                'BackgroundColor', [0.02 0.02 0.02], ...
                'Margin', 2);
        end
    end

    if showLegend && ~isempty(legendHandles)
        lgd = legend(ax, legendHandles, legendLabels, 'Location', 'bestoutside');
        set(lgd, 'TextColor', [0.9 0.9 0.9], 'Color', [0.1 0.1 0.1]);
    end

    if ~isempty(allX)
        pad = 0.05;
        xr = max(allX) - min(allX);
        yr = max(allY) - min(allY);
        if xr == 0, xr = 1; end
        if yr == 0, yr = 1; end
        xlim(ax, [min(allX)-pad*xr, max(allX)+pad*xr]);
        ylim(ax, [min(allY)-pad*yr, max(allY)+pad*yr]);
    end

end


function ids = getNumericIdList(S, fieldName)
    ids = [];

    if ~isfield(S, fieldName)
        return;
    end

    raw = S.(fieldName);
    if isempty(raw)
        return;
    end

    try
        ids = double(raw(:)).';
    catch
        try
            ids = cellfun(@double, raw);
        catch
            ids = [];
        end
    end
end


function jid = getJunctionId(J, fallback)
    if isfield(J, "id")
        jid = double(J.id);
    elseif isfield(J, "junction_id")
        jid = double(J.junction_id);
    else
        jid = fallback;
    end
end


function role = getEdgeRole(E)
    role = "undefined";

    if isfield(E, "role") && ~isempty(E.role)
        role = lower(string(E.role));
    end
end


function lid = getEdgeLinkId(E)
    lid = NaN;

    if isfield(E, "here_link_id") && ~isempty(E.here_link_id)
        lid = double(E.here_link_id);
    elseif isfield(E, "link_id") && ~isempty(E.link_id)
        lid = double(E.link_id);
    end
end


function P = polylineFromGraphEdge(E, nodeMap)
    P = [];

    if isfield(E, "polyline") && ~isempty(E.polyline)
        P = toNx2(E.polyline);
        if ~isempty(P)
            return;
        end
    end

    if isfield(E, "graph_polyline_xy") && ~isempty(E.graph_polyline_xy)
        P = toNx2(E.graph_polyline_xy);
        if ~isempty(P)
            return;
        end
    end

    if ~isfield(E, "u") || ~isfield(E, "v")
        return;
    end

    uKey = idToKey(E.u);
    vKey = idToKey(E.v);

    if isKey(nodeMap, uKey) && isKey(nodeMap, vKey)
        p0 = nodeMap(uKey);
        p1 = nodeMap(vKey);
        if all(isfinite(p0)) && all(isfinite(p1))
            P = [p0; p1];
        end
    end
end


function [nodeMap, nodeXY] = buildNodeMap(G)
    nodeMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    nodeXY = [];

    if ~isfield(G, "nodes") || isempty(G.nodes)
        return;
    end

    N = G.nodes;
    for i = 1:numel(N)
        if ~isfield(N(i), "node_id") || ~isfield(N(i), "x") || ~isfield(N(i), "y")
            continue;
        end

        if isempty(N(i).x) || isempty(N(i).y)
            continue;
        end

        key = idToKey(N(i).node_id);
        xy = [double(N(i).x), double(N(i).y)];

        if ~all(isfinite(xy))
            continue;
        end

        nodeMap(key) = xy;
        nodeXY = [nodeXY; xy]; %#ok<AGROW>
    end
end


function key = idToKey(x)
    if ischar(x) || isstring(x)
        key = char(string(x));
    else
        key = sprintf('%.0f', double(x));
    end
end


function P = toNx2(raw)
    P = [];

    if isempty(raw)
        return;
    end

    if isnumeric(raw)
        A = double(raw);
        if size(A, 2) == 2
            P = A;
        elseif size(A, 1) == 2
            P = A.';
        end
        return;
    end

    if iscell(raw)
        try
            A = cell2mat(raw);
            P = toNx2(A);
            return;
        catch
            return;
        end
    end

    if isstruct(raw)
        if isfield(raw, "x") && isfield(raw, "y")
            x = double(raw.x(:));
            y = double(raw.y(:));
            if numel(x) == numel(y)
                P = [x, y];
            end
        end
    end
end