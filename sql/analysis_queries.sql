
USE turbofan_db;

-- Query 1: Dataset Overview
SELECT 
    dataset,
    split_type,
    COUNT(DISTINCT engine_id) as num_engines,
    MAX(cycle) as max_cycles_seen,
    ROUND(AVG(cycle), 1) as avg_cycle
FROM engine_data
GROUP BY dataset, split_type
ORDER BY dataset, split_type;

-- Query 2: Avg. engine lifespan per dataset
SELECT 
	dataset,
    ROUND(AVG(max_cycle), 1) as avg_lifespan,
    MIN(max_cycle) as shortest_life,
    MAX(max_cycle) as longest_life,
    ROUND(STD(max_cycle), 1) as std_lifespan
FROM engine_rul
GROUP BY dataset
ORDER BY dataset;

-- Query 3: Critical engines (RUL<30) 

SELECT
	dataset,
    COUNT(DISTINCT engine_id) as critical_engines
FROM engine_rul
WHERE RUL < 30
GROUP BY dataset
ORDER BY dataset;

-- Query 4: Model performance comparison 

SELECT
	model_name,
    ROUND(RMSE, 2) as RMSE,
    ROUND(MAE, 2) as MAE,
    ROUND(R2, 4) as R2
FROM model_results
ORDER BY RMSE ASC;

-- Query 5: Engine Health Status Classification
SELECT DISTINCT
    r.dataset,
    r.engine_id,
    r.RUL as remaining_life,
    CASE
        WHEN r.RUL <= 30  THEN 'CRITICAL'
        WHEN r.RUL <= 60  THEN 'WARNING'
        WHEN r.RUL <= 90  THEN 'MONITOR'
        ELSE 'HEALTHY'
    END as health_status
FROM engine_rul r
WHERE r.dataset = 'FD001'
AND r.current_cycle = 100
ORDER BY r.RUL ASC
LIMIT 20;

-- Query 6: Complete fleet health summary at cycle 100
SELECT
    health_status,
    COUNT(*) as num_engines,
    ROUND(AVG(RUL), 1) as avg_RUL
FROM (
    SELECT
        engine_id,
        RUL,
        CASE
            WHEN RUL <= 30  THEN 'CRITICAL'
            WHEN RUL <= 60  THEN 'WARNING'
            WHEN RUL <= 90  THEN 'MONITOR'
            ELSE 'HEALTHY'
        END as health_status
    FROM engine_rul
    WHERE dataset = 'FD001'
    AND current_cycle = 100
) as engine_status
GROUP BY health_status
ORDER BY avg_RUL ASC;

-- Query 7: Fleet health across ALL datasets at cycle 100
SELECT
    dataset,
    health_status,
    COUNT(*) as num_engines,
    ROUND(AVG(RUL), 1) as avg_RUL,
    MIN(RUL) as min_RUL,
    MAX(RUL) as max_RUL
FROM (
    SELECT
        dataset,
        engine_id,
        RUL,
        CASE
            WHEN RUL <= 30  THEN '1-CRITICAL'
            WHEN RUL <= 60  THEN '2-WARNING'
            WHEN RUL <= 90  THEN '3-MONITOR'
            ELSE '4-HEALTHY'
        END as health_status
    FROM engine_rul
    WHERE current_cycle = 100
) as engine_status
GROUP BY dataset, health_status
ORDER BY dataset, health_status;