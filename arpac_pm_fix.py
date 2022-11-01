import argparse
from multiprocessing import Pool, current_process, cpu_count
from unittest import skip
import re
import locale
import humanize
from datatable import fread, iread, rbind, dt, f, update
import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from os import walk
from time import sleep
import numpy as np
import pandas as pd
from tabulate import tabulate


# from dateutil import parser
from time import sleep, time

# from pandas_multiprocess import multi_process

# custom_date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y %H.%M') #parser.parse(x)
required_fields = ["Date", "Time", "CO", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5"]
aditional_fields = [
    "Benzene",
    "Toluene",
    "MPXylen",
    "OXylen",
    "VVento",
    "DVento",
    "TempExt",
    "UMR",
    "RadSN",
    "RadSG",
    "TempInt",
    "UVA",
    "UVB",
    "ALBESUP",
    "ALBEINF",
    "PRESS",
    "PLUV",
    "SO2",
    "",
]


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def timer_func(func):
    # This function shows the execution time of
    """_summary_

    Args:
        func (_type_): _description_

    Returns:
        _type_: _description_
    """ """_summary_

    Args:
        func (_type_): _description_

    Returns:
        _type_: _description_
    """  # the function object passed

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        elapsed = humanize.precisedelta(
            t2 - t1, minimum_unit="microseconds", format="%d"
        )
        print(f"Function {func.__name__!r} executed in {elapsed}")
        return result

    return wrap_func


# def custom_date_parser(x):
#     try:
#         return datetime.strptime(x, "%d/%m/%Y %H.%M")
#     except ValueError:
#         return datetime.strptime(x, "%d/%m/%Y %H:%M")


def convert_df_types(df, strColumTimestamp):
    df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(
        np.float32
    )
    df[strColumTimestamp] = df[strColumTimestamp].map(
        lambda x: datetime.strptime(x, "%d-%b-%Y %H:%M")
    )
    return df


# def convert_types(x, strColumTimestamp):
#     for field in ["CO", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5"]:
#         x[field] = np.float32(x[field])
#     x[strColumTimestamp] = datetime.strptime(x[strColumTimestamp], "%d-%b-%Y %H:%M")
#     return x


def set_month(x, strFile):
    monthStr = re.search(r"\d\d-[a-z]{3,}\d\d", str(strFile))[0].split("-")[1][:-2]
    # allow only three characters for month string
    monthStr = monthStr if len(monthStr) == 3 else monthStr[:3]
    strWithinSearch = x[0].replace("/", "-").replace(".", ":")
    result = re.sub("-\d\d-", "-" + monthStr + "-", strWithinSearch)
    return result


def read_single_file(file, number, total_number):
    locale.setlocale(locale.LC_ALL, "it_IT.utf8")
    frame = fread(
        file,
        fill=True,
        skip_blank_lines=True,
        nthreads=4,
        sep="\t",
        header=False,
        skip_to_line=2,
        # DateTime CO	NO	NO2	NOx	O3	PM10	PM2.5
        # Benzene	Toluene	MPXylen	OXylen	VVento	DVento	TempExt	UMR	RadSN
        # RadSG	TempInt	UVA	UVB	ALBESUP	ALBEINF	PRESS	PLUV	SO2	})
        columns=[dt.str32] + [dt.float64] * 7 + [None] * 20,
    )
    print(f"\nFile {number}/{total_number} - {file}, fixing dates..", end="")
    df_dates = frame[:, 0].to_pandas()
    df_dates = df_dates.apply(set_month, strFile=file, axis=1)
    # print(r)
    frame[:, 0] = df_dates
    print("Done!")
    return frame


@timer_func
def read_all_files(path_arpac_csv):
    number_total_files = sum(1 for _ in Path(path_arpac_csv).rglob("*.csv"))
    files = Path(path_arpac_csv).rglob("*.csv")

    # all_csvs = iread(
    #     tuple(files),
    #     fill=True,
    #     skip_blank_lines=True,
    #     nthreads=4,
    #     sep="\t",
    #     header=False,
    #     skip_to_line=2,
    #     # DateTime CO	NO	NO2	NOx	O3	PM10	PM2.5
    #     # Benzene	Toluene	MPXylen	OXylen	VVento	DVento	TempExt	UMR	RadSN
    #     # RadSG	TempInt	UVA	UVB	ALBESUP	ALBEINF	PRESS	PLUV	SO2	})
    #     columns=[dt.str32] + [dt.float64] * 7 + [None] * 20,
    # )

    all_csvs = [
        read_single_file(filename, n, number_total_files)
        for n, filename in enumerate(files)
    ]
    # combine into one table with rbind
    all_csvs = rbind(all_csvs)
    all_csvs.names = ["DateTime", "CO", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5"]
    return all_csvs, number_total_files


@timer_func
def check_headers(path_arpac_csv):
    files = Path(path_arpac_csv).rglob("*.csv")
    all_test = []
    for file in files:
        with open(file) as csv_file:
            csv_reader = csv.reader(
                csv_file,
                delimiter="\t",
            )
            header = next(csv_reader)
            print(f"Checking file {file}...", end="")
            passed = len(header) == 28 and header == required_fields + aditional_fields
            all_test.append(passed)
            print(
                bcolors.OKGREEN + "PASSED" + bcolors.ENDC
                if passed
                else bcolors.FAIL + "NOT PASSED!!!!" + bcolors.ENDC
            )
    if all(all_test):
        print(bcolors.OKGREEN + "All files passed header check." + bcolors.ENDC)
    else:
        print(
            bcolors.WARNING + "Some files DID NOT passed header check!!" + bcolors.ENDC
        )
        keep_going = input("Are you sure to continue processing [Y/n] ?")
        if keep_going == "Y" or keep_going == "":
            pass
        elif keep_going == "n":
            exit(0)


# Apply to dataframe
@timer_func
def apply_using_multi_processing(df):
    # Divide dataframe to chunks
    num_process = cpu_count() - 1

    chunk_size = int(df.shape[0] / (num_process / 2))  # min(num_process, 6))
    chunks = [
        df.iloc[df.index[i : i + chunk_size]] for i in range(0, df.shape[0], chunk_size)
    ]
    print(
        "Start processing (Dates as valid DateTime, float64->float32) RAW DataFrame using Multiprocessing"
    )
    print(
        f"Number chunks {len(chunks)}, total size DF {chunk_size}, size each chunk {df.shape[0]}"
    )

    # Process dataframes
    with Pool(num_process) as pool:
        results = []
        for c in chunks:
            results.append(pool.apply_async(convert_df_types, (c, "DateTime")))
        output_pool = [result.get() for result in results]
        df_reconstructed = pd.concat(output_pool)
        return df_reconstructed


@timer_func
def convert_df(df):
    df_converted = convert_df_types(df, "DateTime")
    return df_converted


def get_statistics(df, outfile_path):
    listCondMax = [
        ("CO == CO.max()", "CO"),
        ("NO2 == NO2.max()", "NO2"),
        ("O3 == O3.max()", "O3"),
        ("`PM2.5` == `PM2.5`.max()", "PM2.5"),
        ("PM10 == PM10.max()", "PM10"),
        ("NO == NO.max()", "NO"),
        ("NOx == NOx.max()", "NOx"),
    ]
    listCondNeg = [
        ("CO <=0", "CO"),
        ("NO2 <=0", "NO2"),
        ("O3 <=0", "O3"),
        ("`PM2.5` <=0", "PM2.5"),
        ("PM10 <=0", "PM10"),
        ("NO <=0", "NO"),
        ("NOx <=0", "NOx"),
    ]
    cDateStr = datetime.strftime(datetime.now(), "%Y-%B-%d %H:%M:%S")
    out_file_path = outfile_path if outfile_path else f"./output_{cDateStr}.txt"
    with open(out_file_path, "w") as out_file:
        print("\nSummary info on resulting DataFrame")
        print(f"\nReport generated on {cDateStr}", file=out_file)
        print("\nSummary info on resulting DataFrame", file=out_file)
        print(df.info())
        print(df.info(), file=out_file)
        print(df.describe(percentiles=[0, 0.25, 0.5, 0.75, 0.90]))
        print(df.describe(percentiles=[0, 0.25, 0.5, 0.75, 0.90]), file=out_file)
        for s in ["count", "mean", "min", "max"]:
            print(f"\n\nReport for statistic: {s}")
            print(f"\n\nReport for statistic: {s}", file=out_file)
            df_count = df.groupby([df.index.year, df.index.month_name()]).agg(s)
            df_count.index.rename(["Year", "Month"])
            print(df_count)
            print(df_count, file=out_file)

        print("Max values are found at....")
        print("Max values are found at....", file=out_file)
        for cond, field in listCondMax:
            print(df.query(cond)[field].to_frame())
            print(df.query(cond)[field].to_frame(), file=out_file)
        print("\nNegative or zero values are found at....")
        print("\nNegative or zero values are found at....", file=out_file)
        for cond, field in listCondNeg:
            neg = df.query(cond)[field].to_frame()
            num_neg = neg.shape[0]
            percNeg = (num_neg / df.shape[0]) * 100
            print(
                f"{field} has {num_neg} negative or zero values ({percNeg:.3f}% of total)"
            )
            print(
                f"{field} has {num_neg} negative or zero values ({percNeg:.3f}% of total)",
                file=out_file,
            )
            print(neg)
            print(neg, file=out_file)
    return out_file_path


def fix_PM(df, colsPM=["PM2.5", "PM10"]):
    # get DataFrame with PM fields only
    df_PM_only = df[colsPM]
    # find rows with minute==34
    df_PM_only34 = df_PM_only[df_PM_only.index.minute == 34]
    # this result is something as:
    # df_PM_only34
    #                         PM2.5       PM10
    # DateTime
    # 2021-01-13 00:34:00  20.100000  28.100000
    # 2021-01-13 01:34:00  14.600000  18.700001
    # 2021-01-13 02:34:00   9.400000  12.200000
    # 2021-01-13 03:34:00   5.900000   7.800000
    # 2021-01-13 04:34:00   8.800000  14.800000
    # replace minute 34 with minute 0
    df_PM_only34.index = df_PM_only34.index.map(lambda t: t.replace(minute=0))
    # now it is something as
    #     df_PM_only34
    #                         PM2.5       PM10
    # DateTime
    # 2021-01-13 00:00:00  20.100000  28.100000
    # 2021-01-13 01:00:00  14.600000  18.700001
    # 2021-01-13 02:00:00   9.400000  12.200000
    # 2021-01-13 03:00:00   5.900000   7.800000
    # 2021-01-13 04:00:00   8.800000  14.800000
    # shift back of 1 h
    df_PM_only34.index = df_PM_only34.index.shift(-1, freq="H")
    # after shifting becomes
    #     df_PM_only34
    #                         PM2.5       PM10
    # DateTime
    # 2021-01-12 23:00:00  20.100000  28.100000
    # 2021-01-13 00:00:00  14.600000  18.700001
    # 2021-01-13 01:00:00   9.400000  12.200000
    # 2021-01-13 02:00:00   5.900000   7.800000
    # 2021-01-13 03:00:00   8.800000  14.800000

    # sort data on  index and drop rows if nan in both PM2.5 & PM10 if any
    # df_PM_only34.isna().sum() Actually no NaN in PM
    if df_PM_only.isna().sum().values.sum() > 0:
        print(bcolors.WARNING + "Found NaN in PM as reported below")
        print(df_PM_only.isna().sum())
        print(bcolors.ENDC)
    final_df_PM = df_PM_only34.dropna(how="all", axis=0).sort_index()

    # merge data with original DataFrame having removed from this PM columns
    # to avoid _x _y in resulting DataFrame with PM fields duplicated
    df_all_fields = df.drop(columns=colsPM).merge(
        final_df_PM, left_index=True, right_index=True, how="left"
    )
    #                         CO    NO    NO2        NOx         O3      PM2.5    PM10
    # DateTime
    # 2021-01-13 00:00:00  1.10   0.6  31.200001  32.200001  22.400000   14.6  18.700001
    # 2021-01-13 00:01:00  1.09   1.1  30.500000  32.200001  26.200001    NaN        NaN
    # 2021-01-13 00:02:00  1.10   4.4  34.599998  41.299999  22.799999    NaN        NaN
    # 2021-01-13 00:03:00  1.08  11.5  39.500000  57.000000  17.200001    NaN        NaN
    # 2021-01-13 00:57:00  0.84   1.1  23.900000  25.600000  39.000000    NaN        NaN
    # 2021-01-13 00:58:00  0.84   1.0  27.100000  28.700001  36.400002    NaN        NaN
    # 2021-01-13 00:59:00  0.84   0.3  28.500000  29.000000  34.599998    NaN        NaN
    # 2021-01-13 01:00:00  0.83   0.1  24.400000  24.500000  36.799999    9.4  12.200000
    # df_all_fields at PM2.5 and PM10 has many NaN (values only at full hours)
    # fill NaN in PM fields only using ffill
    df_all_fields[colsPM] = df_all_fields[colsPM].fillna(method="ffill")
    #                       CO    NO        NO2        NOx         O3    PM2.5    PM10
    # DateTime
    # 2021-01-13 00:00:00  1.10   0.6  31.200001  32.200001  22.400000   14.6  18.700001
    # 2021-01-13 00:01:00  1.09   1.1  30.500000  32.200001  26.200001   14.6  18.700001
    # 2021-01-13 00:02:00  1.10   4.4  34.599998  41.299999  22.799999   14.6  18.700001
    # 2021-01-13 00:03:00  1.08  11.5  39.500000  57.000000  17.200001   14.6  18.700001
    # 2021-01-13 00:04:00  1.08   9.3  37.200001  51.299999  13.900000   14.6  18.700001
    # 2021-01-13 00:05:00  1.07   4.3  34.900002  41.400002  16.100000   14.6  18.700001
    # 2021-01-13 00:06:00  1.07   3.0  32.299999  36.799999  19.100000   14.6  18.700001
    # 2021-01-13 00:07:00  1.07   2.5  33.200001  36.900002  19.200001   14.6  18.700001
    # 2021-01-13 00:58:00  0.84   1.0  27.100000  28.700001  36.400002   14.6  18.700001
    # 2021-01-13 00:59:00  0.84   0.3  28.500000  29.000000  34.599998   14.6  18.700001
    # 2021-01-13 01:00:00  0.83   0.1  24.400000  24.500000  36.799999    9.4  12.200000
    # 2021-01-13 01:01:00  0.83   0.3  18.200001  18.600000  43.599998    9.4  12.200000
    return df_all_fields


def localize_data(df):
    if df.index.tzinfo == None:
        df.index = df.index.tz_localize(timezone(timedelta(hours=+1)))
    return df


if __name__ == "__main__":

    # Create the parser
    my_parser = argparse.ArgumentParser(
        description="Concantenate ARPAC raw files fixing dates and PM"
    )

    # Add the arguments
    my_parser.add_argument(
        "--path",
        dest=None,
        type=str,
        help="path of root folder containing ARPAC raw files",
    )
    my_parser.add_argument(
        "--output",
        dest=None,
        required=False,
        type=str,
        help="path output statistic file (default ./output_YYYY-B-d H:M:S.txt)",
    )

    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_path = args.path
    out_file = args.output

    if not Path.is_dir(Path(input_path)):
        print(
            bcolors.WARNING
            + "\nThe path specified does not exist or is not a valid directory!!"
            + bcolors.ENDC
        )
        print("Exiting...")
        sleep(2)
        exit(1)

    # FIRST CHECK HEADERS fields
    check_headers(input_path)
    # READ ALL CSV FILES WITHIN PROVIDED DIRECTORY & FIX DATES
    # RETURNING DATA is a DATATABLE (not DataFrame) with fields all float64
    # and only DateTime as string
    all_csv, num_tot_files = read_all_files(input_path)
    # GET A PANDAS DATAFRAMES WITH
    df_all_csv = all_csv.to_pandas()
    print("Info on resulting RAW Dataframe")
    print(tabulate(df_all_csv.info()))
    # CONVERT float64 to float32 and set DateTime as valid timestamp (no TimeZone Aware)
    # using  multiprocessing
    result = apply_using_multi_processing(df_all_csv)
    # DataFrame conversion with no Multiprocessing support
    # uncomment to see difference in speed
    # result = convert_df(df_all_csv)

    # SET DATETIME INDEX
    result = result.set_index("DateTime")
    # SORT DATA
    result = result.sort_index(ascending=True)
    # DROP ROWS HAVING ALL NAN
    result = result.dropna(axis=0, how="all")
    print("Sample from DataFrame having removed NaN and set DateTime index")
    print(result.head())
    # RESULT CHECK
    # result.loc['2021-07-10 12:00']
    # CO        0.400000
    # NO        1.700000
    # NO2       6.800000
    # NOx       9.300000
    # O3       72.300003
    # PM10     22.100000
    # PM2.5    11.300000
    # Name: 2021-07-10 12:00:00, dtype: float32
    # Corresponding file reports
    # Date	            CO	 NO	NO2 NOx  O3	 PM10 PM2.5
    # 10/31/2021 12.00 0.40	1.7	6.8	9.3	72.3 22.1 11.3
    #
    # result.loc['2021-07-10 12:32']
    # CO        0.410000
    # NO        1.000000
    # NO2       5.100000
    # NOx       6.700000
    # O3       65.400002
    # PM10     15.600000
    # PM2.5     6.900000
    # Name: 2021-07-10 12:32:00, dtype: float32
    # Corresponding file reports
    #  Date	            CO	 NO	 NO2  NOx  O3	PM10 PM2.5
    # 10/31/2021 12.32	0.41 1.0 5.1  6.7 65.4	15.6   6.9
    #
    # BORDER VALUE AT MINUTE 31 to remove/discard
    # result.loc['2021-07-10 12:31']
    # CO        0.410000
    # NO        1.000000
    # NO2       5.200000
    # NOx       6.800000
    # O3       66.099998
    # PM10     16.900000
    # PM2.5     7.800000
    # Name: 2021-07-10 12:31:00, dtype: float32
    # Corresponding RAW ARPAC file reports
    #  Date	            CO	 NO	 NO2  NOx  O3	PM10 PM2.5
    # 10/31/2021 12.31	0.41 1.0 5.2  6.8 66.1	16.9  7.8
    #
    # result.loc['2021-07-10 12:30']
    # CO        0.390000
    # NO        1.500000
    # NO2       7.100000
    # NOx       9.300000
    # O3       68.099998
    # PM10     22.100000
    # PM2.5    11.300000
    # Name: 2021-07-10 12:30:00, dtype: float32
    # Corresponding RAW ARPAC file reports
    #  Date	            CO	  NO	NO2  NOx   O3	PM10 PM2.5
    # 10/31/2021 12.30	0.39 1.5	7.1	 9.3  68.1	22.1  11.3

    # Info DataFrame
    print("INFO on DataFrame having removed NaN and set DateTime index")
    result.info()

    # GET STATISTIC AND PRINT RESULT ON CONSOLE AND SAVE TO TXT file
    path_saved_results = get_statistics(result, out_file)
    print(f"\n\nAll done...report file is {path_saved_results}")

    # MAX PM10 value results from DataFrame at
    # DateTime
    # 2021-02-07 15:33:00  1165.699951
    # FROM CORRESPONDING ARPAC RAW FILE results
    #  Date	             CO	  NO	NO2  NOx   O3	PM10    PM2.5
    # 07/29/2021 15.33	0.21 0.6	2.8	 3.7  70.4	1165.7	13.4
    # results DF reports
    # result.loc["2021-02-07 15:33:00"]
    # CO          0.210000
    # NO          0.600000
    # NO2         2.800000
    # NOx         3.700000
    # O3         70.400002
    # PM10     1165.699951
    # PM2.5      13.400000
    # Name: 2021-02-07 15:33:00, dtype: float32
    #
    # MAX NO2 results
    #                          NO2
    # DateTime
    # 2021-08-23 08:30:00  3805.399902
    # Corrsponding file
    #  Date	             CO	    NO	     NO2     NOx     O3	    PM10    PM2.5
    # 23/31/2021 08.30	11.06	13.2	3805.4	3825.5	817.8	35.5	22.7
    # results DF reports
    # result.loc["2021-08-23 08:30"]
    # CO         11.060000
    # NO         13.200000
    # NO2      3805.399902
    # NOx      3825.500000
    # O3        817.799988
    # PM10       35.500000
    # PM2.5      22.700001
    # Name: 2021-08-23 08:30:00, dtype: float32

    # SAVE RESULTS (TZ UNAWARE and NO PM FIX)
    print(
        f"Saving file all CSV ./result_tz_unaware_no_pm_fix.csv (TZ UNAWARE, NO PM FIX)"
    )
    result.reset_index().to_csv("./result_tz_unaware_no_pm_fix.csv", index=False)
    print("Applying PM FIX...")

    # FIX PM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # The measurement cycle of the instrument is this,
    # I will explain it to you so that it is clear why these anomalous minute measurements:
    # a) every hour the instrument loads a filter under the suction duct by passing through it of ambient air,
    #    with consequent accumulation of the PM fraction selected by the suction head outside.
    #    The filter remains on display until minute 59 of the same hour;
    # b) the filter is then moved under the beta ray source with which it is irradiated and a receptor
    #    at the opposite end captures the amount of radiation passing through.
    # c) from the difference between radiation emitted and captured, through an algorithm,
    #    the mass of accumulated particles. By dividing the mass by the flow it then determines the measured
    #    PM concentration.
    # That said, obviously steps b) and c) take some time (about 30 minutes, in fact).
    # It therefore follows that the concentration value that can be read
    # from minute 34 to minute 59 of an hour is actually the measure relating
    # to the accumulation of PM sampled at the previous hour.
    # A practical example: let's consider the time interval 01:00 - 01:59 on 13/01/2021.
    # The value reported from 01:34 to 01:59 (PM10 = 18.7) is actually the average concentration per hour
    # measured from 00:00 to 00:59 (the value 01:33 is completely random, so it should be ignored )...and so on.
    # Obviously this thing is managed at the software level when the hourly averages are consulted
    # while for the minute measurements it is not. The same goes for PM2.5.

    result_PM_fixed = fix_PM(result)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # FROM file of 20 September 2021
    #  Date	             CO	    NO	NO2  NOx     O3	    PM10    PM2.5
    # 20/30/2021 11.28	0.37	2.9	12.1 16.5	53.9	40.0	14.8
    # 20/30/2021 11.29	0.38	3.3	7.3	 12.3	55.3	40.0	14.8
    # 20/30/2021 11.30	0.37	3.0	9.1	 13.8	55.4	40.9	16.3  <<- discard this
    # 20/30/2021 11.31	0.36	2.5	8.0	 11.8	56.4	42.1	18.2
    # 20/30/2021 11.32	0.35	2.7	8.6	 12.7	56.7	42.1	18.2
    # 20/30/2021 11.33	0.36	3.3	10.9 15.9	54.6	42.1	18.2
    # 20/30/2021 11.34	0.38	3.3	11.6 16.7	52.3	42.1	18.2 <<- set these PM values for 2021-09-20 10:00

    # FROM file of 11 January 2022
    #  Date	             CO	    NO	NO2 NOx  O3	    PM10    PM2.5
    # 11/31/2022 00.00	0.36	0.0	7.5	7.5	60.0	18.4	10.2
    # ................  .....   .... .. ... .....   ....    .....
    # 11/31/2022 01.27	0.39	0.1	5.1	5.1	57.9	14.2	11.3
    # 11/31/2022 01.28	0.39	0.2	5.5	5.9	57.4	10.8	5.4 <<- discard this
    # 11/31/2022 01.29	0.39	0.2	6.1	6.3	56.9	10.4	4.7
    # 11/31/2022 01.30	0.37	0.0	6.1	6.0	57.4	10.4	4.7
    # 11/31/2022 01.31	0.35	0.2	5.1	5.4	59.0	10.4	4.7
    # 11/31/2022 01.32	0.37	0.2	5.3	5.7	58.4	10.4	4.7
    # 11/31/2022 01.33	0.38	0.3	5.6	6.0	57.6	10.4	4.7
    # 11/31/2022 01.34	0.39	0.2	5.3	5.6	58.3	10.4	4.7 <<- set these PM values for 2022-01-11 00:00

    # SET DATETIMEINDEX AS UTC+1
    result_PM_fixed_UTC_Plus1 = localize_data(result_PM_fixed)
    print(f"Saving file all CSV ./result_UTC+1_PM_fixed.csv (UTC+1, PM FIXED!!)")
    # save final data as CSV (comma separated, Datetime as UTC+1)
    result_PM_fixed_UTC_Plus1.reset_index().to_csv(
        "./result_UTC+1_PM_fixed.csv", index=False
    )
    # result_PM_fixed_UTC_Plus1.loc["2022-01-11 00:00"]
    # result_PM_fixed_UTC_Plus1.loc["2022-01-11 00:00"]
    # CO        0.36
    # NO        0.00
    # NO2       7.50
    # NOx       7.50
    # O3       60.00
    # PM2.5     4.70
    # PM10     10.40
    # Name: 2022-01-11 00:00:00+01:00, dtype: float32
