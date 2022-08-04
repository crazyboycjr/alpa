#!/usr/bin/env bash

sqlite3 -header -csv results.db 'select * from results where avg is not null or error is not null;' > results_all.csv