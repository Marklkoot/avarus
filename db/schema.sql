CREATE TABLE IF NOT EXISTS portfolio_positions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  coin VARCHAR(20),
  quantity DECIMAL(30,10) NOT NULL DEFAULT 0,
  cost_basis DECIMAL(30,10) NOT NULL DEFAULT 0,
  last_updated DATETIME
);

CREATE TABLE IF NOT EXISTS trades (
  id INT AUTO_INCREMENT PRIMARY KEY,
  coin VARCHAR(20),
  side VARCHAR(10),
  amount DECIMAL(30,10),
  price DECIMAL(30,10),
  fee DECIMAL(30,10),
  timestamp DATETIME
);

CREATE TABLE IF NOT EXISTS meta_parameters (
  id INT AUTO_INCREMENT PRIMARY KEY,
  param_name VARCHAR(100),
  param_value TEXT,
  last_updated DATETIME
);

CREATE TABLE IF NOT EXISTS fundamentals (
  id INT AUTO_INCREMENT PRIMARY KEY,
  coin VARCHAR(20),
  metric_name VARCHAR(50),
  metric_value DECIMAL(30,10),
  date DATE
);

-- For storing daily net worth snapshots:
CREATE TABLE IF NOT EXISTS performance_snapshots (
  id INT AUTO_INCREMENT PRIMARY KEY,
  snapshot_date DATE,
  total_value DECIMAL(30,10),
  note VARCHAR(255)
);

-- For partial fill logic => store open orders
CREATE TABLE IF NOT EXISTS open_orders (
  id INT AUTO_INCREMENT PRIMARY KEY,
  order_id VARCHAR(100),
  coin VARCHAR(20),
  side VARCHAR(10),
  amount DECIMAL(30,10),
  price DECIMAL(30,10),
  filled DECIMAL(30,10) NOT NULL DEFAULT 0,
  status VARCHAR(20),
  created_at DATETIME,
  updated_at DATETIME
);
