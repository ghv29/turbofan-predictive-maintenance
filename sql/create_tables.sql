engine_data-- ================================================
-- Turbofan Predictive Maintenance Database
-- Create Tables Script
-- ================================================

CREATE DATABASE IF NOT EXISTS turbofan_db;
USE turbofan_db;

-- Raw sensor readings table
CREATE TABLE IF NOT EXISTS engine_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    dataset VARCHAR(10) NOT NULL,
    split_type VARCHAR(10) NOT NULL,
    engine_id INT NOT NULL,
    cycle INT NOT NULL,
    op1 FLOAT, op2 FLOAT, op3 FLOAT,
    s1 FLOAT, s2 FLOAT, s3 FLOAT, s4 FLOAT, s5 FLOAT,
    s6 FLOAT, s7 FLOAT, s8 FLOAT, s9 FLOAT, s10 FLOAT,
    s11 FLOAT, s12 FLOAT, s13 FLOAT, s14 FLOAT, s15 FLOAT,
    s16 FLOAT, s17 FLOAT, s18 FLOAT, s19 FLOAT, s20 FLOAT,
    s21 FLOAT
);

-- RUL values table
CREATE TABLE IF NOT EXISTS engine_rul (
    id INT PRIMARY KEY AUTO_INCREMENT,
    dataset VARCHAR(10) NOT NULL,
    engine_id INT NOT NULL,
    max_cycle INT NOT NULL,
    current_cycle INT NOT NULL,
    RUL INT NOT NULL
);

-- Model results table
CREATE TABLE IF NOT EXISTS model_results (
    id INT PRIMARY KEY AUTO_INCREMENT,
    dataset VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    RMSE FLOAT NOT NULL,
    MAE FLOAT NOT NULL,
    R2 FLOAT NOT NULL,
    run_date DATETIME NOT NULL
);

