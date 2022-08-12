# Codesign toolkit for ZHEN

```
$ python3 codesign/main.py --help
usage: main.py [-h] [-D] [-c CONFIG] [-d DB] [--search-model] [--manual-job-timeout MANUAL_JOB_TIMEOUT] [--retry-failed]

ZHEN codesign toolkit.

optional arguments:
  -h, --help            show this help message and exit
  -D, --dry-run         Do not run. Only print what settings was chosen to run. (default: False)
  -c CONFIG, --config CONFIG
                        The path to the configuration. (default: codesign/models/config.toml)
  -d DB, --db DB        The path to the sqlite3 database file. All the results will be saved there. (default:
                        codesign/results.db)
  --search-model        Whether to search the model. If not, use the models specified in the config.toml. (default: False)
  --manual-job-timeout MANUAL_JOB_TIMEOUT
                        The timeout threshold in seconds for jobs using manual stage. (Default: 10min) (default: 600)
  --retry-failed        Whether to retry failed jobs in the results database. (default: False)
```

## Motivation
There is little low-hanging fruit in improving DHEN’s training performance for single and multiple GPU scenario. However, both macro and micro architecture affect the performance even if the parameter count remain the same. To further exploit DHEN’s training performance, we seek to model and software codesign. Taking advantage of Alpa, a state-of-the-art 3D parallelism runtime, we can explore the design space of DHEN in regard to performance through automated search.

This codesign toolkit is our first step towards this automated
architecture search.


## Define and genrate model search space
To search for model specifications, run
```
python3 codesign/main.py --search-model
```
This will generate a list of model specs and print them to stdout. You
can manually copy or redirect the output to a file to create the
configuration. Please look at `models/config.toml` for an example of
configuration file. See `search.py` for more details about enumerating
and filtering the model candidates. To change the filtering criteria or
enlarging the search space, you will need to modify this file.

## Enumerate training execution space
The enumration is implemented in `main.py` and `config.py`. In short,
for each `model`, it enumerates all the 'matched' combination of
`cluster_spec` and `parallel_spec` and run the model and with the
execution configuration (A `cluster_spec` and a `parallel_spec` do not match
when they target for different number of GPUs.).


## Run
First, edit `run.sh` to let it find your custom configuration. Then,
```
./run.sh
```
This script will execute the toolkit and retry the toolkit when
exceptions detected. The toolkit will record the attempted
candidate configuration by writing a database, so it will automatically
skip failed records.  You can also specify the `--retry-failed` option
to make it retry the enumerated but once failed configurations.

## Collect the result
All the specifications, runtime/global configurations, environment
variables, results, and errors are recorded in a SQLite database.
You can use `sqlite3 results.db` to connect to the database and use SQL
commands to inspect it. It is recommended to [DB Browser for
SQLite](https://sqlitebrowser.org/) which I personally found very
convenient to browse and edit the record (e.g., deleting a particular
list of failed jobs so that you can rerun your script to retry those jobs).

You can alos use `export_all.sh` or `export_succeded.sh` to export the
database to csv which you can upload to online spreadsheet for processing
and sharing.
