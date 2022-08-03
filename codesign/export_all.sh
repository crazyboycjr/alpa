#!/usr/bin/env bash

sqlite3 -header -csv results.db 'select * from results;' > results_all.csv