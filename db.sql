USE dbname;
USE sql5777489; (EG)
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    otp VARCHAR(6),
    email VARCHAR(255) NOT NULL;
);

INSERT INTO users (username, password, email) VALUES
('swetha','swetha','sswetha0205@gmail.com')