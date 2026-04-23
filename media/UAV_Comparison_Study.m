function UAV_Comparison_Study
    % UAV_Comparison_Study_v3
    % FIXES: 'horzcat' dimension error in initialization.
    % ADDRESSES: All previous feedback (Baselines, Safety Margins, Wind, Convergence).
    
    clc; clear; close all;

    %% 1. Configuration & Environment
    config.map_size = [500, 500, 100]; 
    config.start = [10, 10, 0];
    config.target = [480, 480, 20];
    config.n_control_points = 15; 
    
    config.alpha = 1.0; config.beta = 15.0; config.gamma = 100000;
    config.safety_margin = 3.0;
    
    config.wind.vector = [-8, -4, 0]; 
    config.wind.effect_strength = 0.7;
    config.energy.base_cost = 100;
    config.energy.vert_coeff = 2.5; 
    config.energy.initial_SoC = 5000;

    rng(42); % Fixed seed for fair comparison
    fprintf('Generating Procedural City...\n');
    config.buildings = generate_city_grid(config.map_size);

    %% 2. Run Comparative Study
    modes = {'GA_Only', 'PSO_Only', 'Hybrid'};
    results = struct();
    
    n_vars = config.n_control_points * 3;
    lb = repmat([0, 0, 0], 1, config.n_control_points);
    ub = repmat(config.map_size, 1, config.n_control_points);
    pop_size = 100;
    
    % Generate population using the FIXED function
    initial_pop = generate_smart_init(pop_size, config, lb, ub);

    for m = 1:length(modes)
        mode = modes{m};
        fprintf('\n--- Running Mode: %s ---\n', mode);
        
        tic;
        global history_log; 
        history_log = []; % Reset history
        
        % -- Optimization Logic --
        if strcmp(mode, 'GA_Only')
            opts = optimoptions('ga', 'Display', 'off', 'MaxGenerations', 100, ...
                'PopulationSize', pop_size, 'InitialPopulationMatrix', initial_pop, ...
                'OutputFcn', @save_history_ga);
            
            [sol, cost] = ga(@(x) fitness_function(x, config), n_vars, [],[],[],[], lb, ub, [], opts);
            
        elseif strcmp(mode, 'PSO_Only')
            opts = optimoptions('particleswarm', 'Display', 'off', 'MaxIterations', 100, ...
                'SwarmSize', pop_size, 'InitialSwarmMatrix', initial_pop, ...
                'OutputFcn', @save_history_pso);
            
            [sol, cost] = particleswarm(@(x) fitness_function(x, config), n_vars, lb, ub, opts);
            
        elseif strcmp(mode, 'Hybrid')
            % Stage 1: GA
            opts_ga = optimoptions('ga', 'Display', 'off', 'MaxGenerations', 40, ...
                'PopulationSize', pop_size, 'InitialPopulationMatrix', initial_pop, ...
                'OutputFcn', @save_history_ga);
            
            [~, ~, ~, ~, final_pop, ~] = ga(@(x) fitness_function(x, config), n_vars, [],[],[],[], lb, ub, [], opts_ga);
            
            % Stage 2: PSO
            opts_pso = optimoptions('particleswarm', 'Display', 'off', 'MaxIterations', 60, ...
                'SwarmSize', pop_size, 'InitialSwarmMatrix', final_pop, ...
                'OutputFcn', @save_history_pso);
            
            [sol, cost] = particleswarm(@(x) fitness_function(x, config), n_vars, lb, ub, opts_pso);
        end
        
        time_taken = toc;
        
        % Save Results
        [path, stats] = reconstruct_path_spline(sol, config);
        results(m).mode = mode;
        results(m).cost = cost;
        results(m).time = time_taken;
        results(m).dist = stats.dist_total;
        results(m).energy = stats.energy_total;
        results(m).smoothness = stats.smoothness_score;
        results(m).collisions = stats.collisions;
        results(m).path = path;
        results(m).energy_profile = stats.energy_profile;
        results(m).wind_factors = stats.wind_factors;
        results(m).convergence = history_log; 
    end

    %% 3. Visualization
    hybrid_idx = 3; 
    
    % --- Figure 1: 3D Trajectory ---
    figure('Name', 'Fig 1: Hybrid Trajectory', 'Color', 'w', 'Position', [50 50 1200 600]);
    subplot(1, 2, 1); hold on; axis equal; grid on; view(3);
    title('Hybrid GA-PSO Trajectory with Safety Margins');
    xlim([0 500]); ylim([0 500]); zlim([0 100]);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    
    for i = 1:size(config.buildings, 1)
        draw_building_with_margin(config.buildings(i,:), config.safety_margin);
    end
    
    path = results(hybrid_idx).path;
    surface([path(:,1), path(:,1)], [path(:,2), path(:,2)], [path(:,3), path(:,3)], ...
            [path(:,3), path(:,3)], 'FaceColor', 'no', 'EdgeColor', 'interp', 'LineWidth', 2);
    colormap(gca, turbo); colorbar;
    plot3(config.start(1), config.start(2), config.start(3), 'gs', 'MarkerSize',10,'MarkerFaceColor','g');
    plot3(config.target(1), config.target(2), config.target(3), 'rs', 'MarkerSize',10,'MarkerFaceColor','r');

    % --- Figure 2: Wind & Energy Analysis ---
    subplot(1, 2, 2); hold on; grid on;
    title('Energy Consumption vs Wind Impact');
    
    % Plot Cumulative Energy (Left Axis)
    yyaxis left
    plot(results(hybrid_idx).energy_profile, 'r-', 'LineWidth', 2);
    ylabel('Cumulative Energy (Joules)', 'Color', 'r');
    xlabel('Path Steps');
    
    % Plot Wind Factor (Right Axis)
    yyaxis right
    w_factors = results(hybrid_idx).wind_factors;
    plot(w_factors, 'b-', 'LineWidth', 1.5);
    ylabel('Wind Penalty Factor (>1 = Headwind)', 'Color', 'b');
    yline(1, 'k--'); % Neutral wind line
    
    % Mark Headwind Zones
    headwind_mask = w_factors > 1.05;
    if any(headwind_mask)
        x_zone = 1:length(w_factors);
        % Fill area visually
        area(x_zone(headwind_mask), w_factors(headwind_mask), 'FaceColor', 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        text(length(w_factors)/2, 1.2, 'Blue Shade = Headwind Zone', 'Color', 'b', 'FontSize', 8);
    end

    % --- Figure 3: Convergence & Metrics ---
    figure('Name', 'Fig 2: Convergence & Metrics', 'Color', 'w', 'Position', [100 100 1000 500]);
    
    % Convergence Plot
    subplot(1, 2, 1); hold on; grid on;
    for m = 1:length(modes)
        plot(results(m).convergence, 'LineWidth', 2, 'DisplayName', modes{m});
    end
    legend; xlabel('Iteration'); ylabel('Best Cost'); title('Convergence Comparison');
    
    % Metrics Bar Chart
    subplot(1, 2, 2);
    vals = [[results.energy]; [results.smoothness] .* 100]'; % Scale smoothness for visibility
    b = bar(vals);
    set(gca, 'XTickLabel', modes);
    legend({'Energy (J)', 'Smoothness Score (x100)'});
    title('Quantitative Metrics');
    grid on;

    % --- Console Output Table ---
    fprintf('\n================ FINAL COMPARATIVE REPORT ================\n');
    fprintf('%-10s | %-10s | %-10s | %-12s | %-10s\n', 'Mode', 'Time(s)', 'Energy(J)', 'Smoothness', 'Collisions');
    fprintf('------------------------------------------------------------------\n');
    for m = 1:length(modes)
        fprintf('%-10s | %-10.2f | %-10.0f | %-12.2f | %-10d\n', ...
            results(m).mode, results(m).time, results(m).energy, results(m).smoothness, results(m).collisions);
    end
end

%% --- Helper Functions ---

% --- OUTPUT FUNCTION FOR GA ---
function [state, options, optchanged] = save_history_ga(options, state, flag)
    global history_log;
    optchanged = false;
    if strcmp(flag, 'iter')
        current_best = min(state.Score); 
        history_log = [history_log; current_best];
    end
end

% --- OUTPUT FUNCTION FOR PSO ---
function stop = save_history_pso(optimValues, state)
    global history_log;
    stop = false;
    if strcmp(state, 'iter')
        history_log = [history_log; optimValues.bestfval];
    end
end

% --- FIXED INITIALIZATION FUNCTION ---
function pop = generate_smart_init(pop_size, config, lb, ub)
    n_vars = length(lb); pop = zeros(pop_size, n_vars);
    progress_vals = linspace(0, 1, config.n_control_points + 2);
    baseline_vec = progress_vals(2:end-1); 
    
    for i = 1:pop_size
        base_pts = config.start + (config.target - config.start) .* baseline_vec';
        
        % Taper factor is a COLUMN (15x1)
        taper_factor = sin(pi * baseline_vec'); 
        
        % FIX: Concatenate as [COL, COL, COL] - No transpose needed on taper_factor
        taper_matrix = [taper_factor, taper_factor, ones(config.n_control_points,1)];
        
        noise = (rand(config.n_control_points, 3) - 0.5) * 200 .* taper_matrix;
        
        % Z-axis noise separate scaling
        noise(:,3) = (rand(config.n_control_points, 1) * 60) .* taper_factor; 
        
        guess_flat = reshape(base_pts + noise, 1, []);
        pop(i, :) = max(min(guess_flat, ub), lb);
    end
end

function [smooth_path, stats] = reconstruct_path_spline(vars, config)
    % Extract Control Points
    ctrl_pts = reshape(vars, [config.n_control_points, 3]);
    
    % --- FIX START: PREVENT LOOPS/KNOTS ---
    % 1. Calculate Progress Projection
    vec_st = config.target - config.start;
    len_sq = norm(vec_st)^2;
    
    % Project points onto the Start->Target line (value 0.0 to 1.0)
    projections = (ctrl_pts - config.start) * vec_st' / len_sq;
    
    % 2. FILTER "BACKWARD" POINTS
    % Remove any point that is behind Start (< 0.05) or past Target (> 0.95)
    % This deletes the bad points causing the "knot" loop.
    valid_mask = projections > 0.05 & projections < 0.95;
    
    ctrl_pts = ctrl_pts(valid_mask, :);
    projections = projections(valid_mask);
    
    % 3. SORT
    % Order the remaining points by their progress
    [~, sort_idx] = sort(projections);
    ctrl_pts = ctrl_pts(sort_idx, :);
    % --- FIX END ---
    
    % Create Path Waypoints
    waypoints = [config.start; config.start; ctrl_pts; config.target; config.target];
    
    % Generate Spline (Flyable Curve)
    if size(waypoints, 1) < 4
        % Fallback if too many points were removed (rare): straight line
        t_vals = linspace(0, 1, size(waypoints,1));
        t_query = linspace(0, 1, 200);
        smooth_path = interp1(t_vals, waypoints, t_query, 'linear');
    else
        t_vals = linspace(0, 1, size(waypoints, 1));
        t_query = linspace(0, 1, 200); 
        px = spline(t_vals, waypoints(:,1), t_query);
        py = spline(t_vals, waypoints(:,2), t_query);
        pz = spline(t_vals, waypoints(:,3), t_query);
        smooth_path = [px', py', pz'];
    end
    
    % --- Physics & Collision Check ---
    dist_total = 0; energy_total = config.energy.base_cost; collisions = 0;
    energy_profile = zeros(length(smooth_path), 1);
    wind_factors = zeros(length(smooth_path), 1);
    smoothness_score = 0;
    wind_vec = config.wind.vector; wind_mag = norm(wind_vec);
    
    for i = 1:length(smooth_path)-1
        p1 = smooth_path(i,:); p2 = smooth_path(i+1,:);
        vec = p2 - p1; dist = norm(vec);
        dist_total = dist_total + dist;
        
        % Smoothness
        if i > 1
             prev_vec = smooth_path(i,:) - smooth_path(i-1,:);
             if norm(vec)*norm(prev_vec) > 0
                cos_angle = dot(vec, prev_vec) / (norm(vec)*norm(prev_vec));
                smoothness_score = smoothness_score + (1 - cos_angle);
             end
        end

        d_horiz = norm(p2(1:2) - p1(1:2)); d_vert = abs(p2(3) - p1(3));
        cos_theta = 0;
        if wind_mag > 0 && dist > 0, cos_theta = dot(vec, wind_vec) / (dist * wind_mag); end
        wind_factor = 1 - (config.wind.effect_strength * cos_theta);
        wind_factors(i) = wind_factor;
        
        energy_total = energy_total + (d_horiz + config.energy.vert_coeff * d_vert) * wind_factor;
        energy_profile(i+1) = energy_total;
        
        if check_collision_point(p2, config.buildings, config.safety_margin), collisions = collisions + 1; end
    end
    stats.dist_total = dist_total; stats.energy_total = energy_total;
    stats.collisions = collisions; stats.energy_profile = energy_profile;
    stats.wind_factors = wind_factors; stats.smoothness_score = smoothness_score;
end

function cost = fitness_function(vars, config)
    [~, stats] = reconstruct_path_spline(vars, config);
    cost = (config.alpha * stats.dist_total) + (config.beta * stats.energy_total) + (config.gamma * stats.collisions);
end

function is_col = check_collision_point(pt, buildings, margin)
    is_col = false; x=pt(1); y=pt(2); z=pt(3);
    if z < 0, is_col = true; return; end
    for i = 1:size(buildings, 1)
        b = buildings(i,:);
        if x >= (b(1)-b(3)/2-margin) && x <= (b(1)+b(3)/2+margin) && ...
           y >= (b(2)-b(4)/2-margin) && y <= (b(2)+b(4)/2+margin) && z <= (b(5)+margin)
            is_col = true; return;
        end
    end
end

function draw_building_with_margin(b, margin)
    x_min = b(1) - b(3)/2; x_max = b(1) + b(3)/2;
    y_min = b(2) - b(4)/2; y_max = b(2) + b(4)/2;
    z_max = b(5);
    verts = [x_min y_min 0; x_max y_min 0; x_max y_max 0; x_min y_max 0; ...
             x_min y_min z_max; x_max y_min z_max; x_max y_max z_max; x_min y_max z_max];
    faces = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
    col = 0.9 - (z_max/100)*0.7; 
    patch('Vertices', verts, 'Faces', faces, 'FaceColor', [col col col], 'FaceAlpha', 0.8, 'EdgeColor', 'none');
    
    % Draw Wireframe Margin
    mx_min = x_min - margin; mx_max = x_max + margin;
    my_min = y_min - margin; my_max = y_max + margin; mz_max = z_max + margin;
    plot3([mx_min, mx_max, mx_max, mx_min, mx_min], [my_min, my_min, my_max, my_max, my_min], [0, 0, 0, 0, 0], 'r:', 'LineWidth', 0.5);
    plot3([mx_min, mx_max, mx_max, mx_min, mx_min], [my_min, my_min, my_max, my_max, my_min], [mz_max, mz_max, mz_max, mz_max, mz_max], 'r:', 'LineWidth', 0.5);
end

function buildings = generate_city_grid(map_size)
    buildings = [];
    for x = 40:60:(map_size(1)-40)
        for y = 40:60:(map_size(2)-40)
            if rand > 0.2 
                r = rand;
                if r > 0.7, w=15+rand*10; d=15+rand*10; h=60+rand*30; 
                elseif r > 0.3, w=25+rand*15; d=25+rand*15; h=30+rand*20; 
                else, w=40+rand*10; d=40+rand*10; h=10+rand*15; end
                buildings = [buildings; x+(rand-0.5)*10, y+(rand-0.5)*10, w, d, h];
            end
        end
    end
end