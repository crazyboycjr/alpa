#!/usr/bin/env bash

sqlite3 -header -csv results.db 'select * from results where avg is not NULL;' > results_succeeded.csv