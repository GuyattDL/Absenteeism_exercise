DROP DATABASE IF EXISTS predicted_outputs;

CREATE DATABASE IF NOT EXISTS predicted_outputs;

USE predicted_outputs;

DROP TABLE IF EXISTS predicted_outputs;

CREATE TABLE predicted_outputs
(
	reason_1 bit not null,
	reason_2 bit not null,
	reason_3 bit not null,
	reason_4 bit not null,
	weekday int not null,
	transportation_expense int not null,
	age int not null,
	body_mass_index int not null,
	education bit not null,
	children int not null,
	pets int not null,
	probability float not null,
	prediction bit not null
);

SELECT
*
from
predicted_outputs;