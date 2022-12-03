import TimeTagger
import numpy as np
import numba
import math
from time import sleep


"""
Modified from the example code provided by swabian to generate histgrams from a phased locked clock.
Andrew Mueller February 2022

"""


class CustomPLLHistogram(TimeTagger.CustomMeasurement):
    """
    Example for a single start - multiple stop measurement.
        The class shows how to access the raw time-tag stream.
    """

    def __init__(
        self,
        tagger,
        data_channels,
        clock_channel,
        mult=1,
        phase=0,
        deriv=0.01,
        prop=2e-9,
        n_bins=20000000,
    ):
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        self.data_channels = np.array(data_channels)
        self.clock_channel = clock_channel
        self.mult = mult
        self.phase = phase
        self.deriv = deriv
        self.prop = prop
        self.clock0 = 0
        self.raw_clock = 0
        self.period = 1  # 12227788.110837
        self.phi_old = 0
        self.init = 1
        self.max_bins = n_bins

        self.clock_idx = 0
        self.hist_idxs = np.array([0, 0])
        self.slope_diffs_idx = 0
        self.coinc_idx = 0
        self.old_clock_start = 0
        self.old_clock = 0
        self.i = 1
        for chan in self.data_channels:
            self.register_channel(channel=int(chan))
        self.register_channel(channel=clock_channel)
        self.clear_impl()

        # At the end of a CustomMeasurement construction,
        # we must indicate that we have finished.
        self.finalize_init()

    def __del__(self):
        # The measurement must be stopped before deconstruction to avoid
        # concurrent process() calls.
        self.stop()

    def getData(self):
        # Acquire a lock this instance to guarantee that process() is not running in parallel
        # This ensures to return a consistent data.

        # clocks = np.zeros(50)
        # pclocks = np.zeros(50)
        # # hist_1_tags = np.zeros(50)
        # # hist_2_tags = np.zeros(50)  # why do I make these?
        # self.hist_idxs = np.zeros(len(self.data_channels), dtype=np.int64)
        # self.hist_tags_data = np.zeros(
        #     (
        #         50,
        #         self.max_bins,
        #     ),
        #     dtype=np.float64,
        # )

        while 1:
            self._lock()
            if self.clock_idx == 0:
                self._unlock()
                continue
            if (self.old_clock_start != self.clock_data[0]) | (
                self.old_clock_start == 0
            ):
                clocks = self.clock_data[: self.clock_idx].copy()
                pclocks = self.lclock_data[: self.clock_idx].copy()
                hist_tags = []
                for i, chan in enumerate(self.data_channels):
                    hist_tags.append(self.hist_tags_data[i, : self.hist_idxs[i]].copy())

                slope_diffs = self.slope_diffs[: self.slope_diffs_idx].copy()

                coinc = self.coinc[: self.coinc_idx].copy()
                self.old_clock_start = self.clock_data[0]

                # expiremental ####
                self.clock_idx = 0
                self.hist_idxs = np.zeros(len(self.data_channels), dtype=np.int64)
                self.slope_diffs_idx = 0
                self.coinc_idx = 0
                print("stats 0r: ", self.stats[0] / np.sum(self.stats))
                print("stats 1r: ", self.stats[1] / np.sum(self.stats))
                print("stats 2+r: ", self.stats[2] / np.sum(self.stats))
                print()
                print("stats 0r: ", self.stats[0])
                print("stats 1r: ", self.stats[1])
                print("stats 2+r: ", self.stats[2])
                print("##########################")

                self.stats = np.zeros(3, dtype=np.int64)
                ###################
                self._unlock()
                return clocks, pclocks, hist_tags, slope_diffs
            else:
                print("nope")
            self._unlock()

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self.last_start_timestamp = 0
        self.clock_data = np.zeros((self.max_bins,), dtype=np.int64)
        self.lclock_data = np.zeros((self.max_bins,), dtype=np.int64)
        self.lclock_data_dec = np.zeros(
            (self.max_bins,), dtype=np.float64
        )  # decimal component of clock0
        self.hist_tags_data = np.zeros(
            (
                len(self.data_channels),
                self.max_bins,
            ),
            dtype=np.float64,
        )
        self.coinc = np.zeros((self.max_bins,), dtype=np.float64)
        self.hist_2_tags_data = np.zeros((self.max_bins,), dtype=np.float64)
        self.slope_diffs = np.zeros((self.max_bins,), dtype=np.float64)

        self.raw_buffer = np.zeros((len(self.data_channels), 1), dtype=np.int64)
        self.hist_buffer = np.zeros((len(self.data_channels), 1), dtype=np.float64)
        self.stats = np.zeros(3, dtype=np.int64)

    def on_start(self):
        # The lock is already acquired within the backend.
        pass

    def on_stop(self):
        # The lock is already acquired within the backend.
        pass

    # I should support the measurment of the unfiltered clock with respect to the phase locked clock.

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def fast_process(
        tags,
        clock_data,
        lclock_data,
        lclock_data_dec,
        hist_tags_data,
        slope_diffs,
        coinc,
        data_channels,
        clock_channel,
        init,
        clock0,
        raw_clock,
        period,
        phi_old,
        deriv,
        prop,
        phase,
        mult,
        clock_idx,
        hist_idxs,
        slope_diffs_idx,
        coinc_idx,
        q,
        raw_buffer,
        hist_buffer,
        stats,
    ):

        """
        A precompiled version of the histogram algorithm for better performance
        nopython=True: Only a subset of the python syntax is supported.
                       Avoid everything but primitives and numpy arrays.
                       All slow operation will yield an exception
        nogil=True:    This method will release the global interpreter lock. So
                       this method can run in parallel with other python code
        """

        zero_cycles = 0
        empty_time = 180000
        tag_1_buffer = 0
        tag_2_buffer = 0
        q = 0
        zero_kets = 0
        plus_1_kets = 0

        if init:
            msg = f"Init PLL with clock channel {clock_channel}"
            for i, channel in enumerate(data_channels):
                msg = msg + f", data channel {i}: {channel}"
            print(msg)

            clock_idx = 0
            clock_portion = np.zeros(1000, dtype=np.uint64)

            for clock_idx, tag in enumerate(tags[:1000]):
                if tag["channel"] == clock_channel:
                    clock_portion[clock_idx] = tag["time"]
                    clock_idx += 1

            # Initial Estimates
            clock_portion = clock_portion[clock_portion > 0]  # cut off extra zeros
            period = (clock_portion[-1] - clock_portion[0]) / (len(clock_portion) - 1)
            # freq = 1 / period
            clock0 = -1
            clock0_dec = -0.1
            print("[READY] Finished FastProcess Initialization")
            clock_idx = 0
            hist_idxs = np.zeros(len(data_channels), dtype=np.int64)
            coinc_idx = 0

        for tag in tags:
            q = q + 1

            if tag["channel"] == clock_channel:
                clock0, period, clock_data, clock_idx, phi_old = clock_lock(
                    tag["time"],
                    clock_data,
                    lclock_data,
                    lclock_data_dec,
                    clock0,
                    period,
                    phi_old,
                    clock_idx,
                    prop,
                    deriv,
                )

            if (tag["channel"] == data_channels[0]) or (
                tag["channel"] == data_channels[1]
            ):
                if clock0 == -1:
                    continue

                for i, data_chan in enumerate(data_channels):
                    if tag["channel"] == data_chan:
                        prev_raw_tag = raw_buffer[i, 0]

                        delta_time = tag["time"] - prev_raw_tag
                        if delta_time > empty_time:

                            cycles = round(
                                delta_time / 100000, 1
                            )  # about 200, 300, 400, etc

                            # I need to be careful about double counting
                            stats[0] += int((cycles - 2.0) / 2)
                            # if cycles == 2.0:
                            #     stats[1] += 1

                            hist_tag = ((tag["time"]) - clock0) - clock0_dec
                            sub_period = period / mult
                            minor_cycles = (hist_tag + phase) // sub_period
                            hist_tag = hist_tag - (sub_period * minor_cycles)
                            hist_tags_data[i, hist_idxs[i]] = hist_tag
                            hist_idxs[i] += 1
                            save_to_buffer(hist_buffer, i, hist_tag)

                            if i == 0:
                                j = 1
                            else:
                                j = 0
                            prev_other_channel = raw_buffer[j, 0]
                            prev_other_channel_hist = hist_buffer[j, 0]
                            # if the raw tags are very close in time
                            if abs(tag["time"] - prev_other_channel) < 2000:
                                # then average the clock-referenced hist tags
                                diff = (hist_tag + prev_other_channel_hist) / 2
                                slope_diffs[slope_diffs_idx] = diff
                                slope_diffs_idx += 1
                                if diff < 236:
                                    stats[2] += 1
                                else:
                                    stats[1] += 1

                        save_to_buffer(raw_buffer, i, tag["time"])
                        """
                        need some way of loading differnt features here, 
                        and exporting their resutls in an orderly way
                        """

                    # else:
                    #     continue

                # if (tag["channel"] == data_channel_1) or (tag["channel"] == data_channel_2):
                #     if clock0 != -1:
                #         hist_tag = ((tag["time"]) - clock0) - clock0_dec
                #         # hist_tag = tag["time"] - raw_clock # NO pll
                #         # hist_tag = (tag["time"]+test_factor) - current_clock # no PLL
                #         sub_period = period / mult
                #         minor_cycles = (hist_tag + phase) // sub_period
                #         hist_tag = hist_tag - (sub_period * minor_cycles)

                #     # if no previous tag for empty_time ealier (no jitterate)
                #     if tag["time"] - old_tag_1 > empty_time:
                #         hist_1_tags_data[hist_1_idx] = hist_tag
                #         hist_1_idx += 1
                #         tag_1_buffer = tag["time"]
                #     old_tag_1 = tag["time"]

                #     if abs(tag["time"] - tag_2_buffer) < 2000:
                #         diff = tag["time"] - tag_2_buffer
                #         slope_diffs[slope_diffs_idx] = diff
                #         slope_diffs_idx += 1

                # if tag["channel"] == data_channel_2:
                #     if tag["time"] - old_tag_2 > empty_time:

                #         hist_2_tags_data[hist_2_idx] = hist_tag
                #         hist_2_idx += 1
                #         tag_2_buffer = tag["time"]
                #     old_tag_2 = tag["time"]

                #     if abs(tag["time"] - tag_1_buffer) < 2000:
                #         diff = tag["time"] - tag_1_buffer
                #         slope_diffs[slope_diffs_idx] = diff
                #         slope_diffs_idx += 1

                # hist_2_tags_data[hist_2_idx] = hist_tag
                # hist_2_idx += 1
        # print("zero_kets: ", zero_kets)
        # print("one_plus_kets: ", plus_1_kets)
        # print("zero_ratio: ", zero_kets / (zero_kets + plus_1_kets))
        # print("one_plus_ratio: ", plus_1_kets / (zero_kets + plus_1_kets))
        # print("#####################")

        if init:
            init = 0

        return (
            clock0,
            raw_clock,
            period,
            phi_old,
            init,
            clock_idx,
            hist_idxs,
            slope_diffs_idx,
            coinc_idx,
            q,
        )

    def process(self, incoming_tags, begin_time, end_time):
        """
        Main processing method for the incoming raw time-tags.

        The lock is already acquired within the backend.
        self.data is provided as reference, so it must not be accessed
        anywhere else without locking the mutex.

        Parameters
        ----------
        incoming_tags
            The incoming raw time tag stream provided as a read-only reference.
            The storage will be deallocated after this call, so you must not store a reference to
            this object. Make a copy instead.
            Please note that the time tag stream of all channels is passed to the process method,
            not only the onces from register_channel(...).
        begin_time
            Begin timestamp of the of the current data block.
        end_time
            End timestamp of the of the current data block.
        """
        (
            self.clock0,
            self.raw_clock,
            self.period,
            self.phi_old,
            self.init,
            self.clock_idx,
            self.hist_idxs,
            self.slope_diffs_idx,
            self.coinc_idx,
            self.i,
        ) = CustomPLLHistogram.fast_process(
            incoming_tags,
            self.clock_data,
            self.lclock_data,
            self.lclock_data_dec,
            self.hist_tags_data,
            self.slope_diffs,
            self.coinc,
            self.data_channels,
            self.clock_channel,
            self.init,
            self.clock0,
            self.raw_clock,
            self.period,
            self.phi_old,
            self.deriv,
            self.prop,
            self.phase,
            self.mult,
            # expiremental
            self.clock_idx,
            self.hist_idxs,
            self.slope_diffs_idx,
            self.coinc_idx,
            self.i,
            self.raw_buffer,
            self.hist_buffer,
            self.stats,
        )


# buffer = np.zeros(channels, length)


@numba.jit(nopython=True, nogil=True, cache=True)
def save_to_buffer(buffer, channel_idx, tag):
    # saves a new tag to the top of the buffer for a particular channel
    # transalte everything in channel down
    buffer[channel_idx, 1:] = buffer[channel_idx, :-1]
    buffer[channel_idx, 0] = tag


@numba.jit(nopython=True, nogil=True, cache=True)
def get_from_buffer(buffer, channel_idx, loc):
    return buffer[channel_idx, loc]


@numba.jit(nopython=True, nogil=True, cache=True)
def clock_lock(
    current_clock,
    clock_data,
    lclock_data,
    lclock_data_dec,
    clock0,
    period,
    phi_old,
    clock_idx,
    prop,
    deriv,
):
    """
    at the end of the function, clock0 is the 'locked' version of the current
    incoming clock (current_clock)

    care must be taken to handle int64s, and not convert them to floats if they
    are expected to be very large (like the raw time in picoseconds).

    clock0_dec is the decimal remainder (a float) for clock0.

    period is the 'pll loop constant', or, the thing that updates slowly in response
    to phase error measurments.

    """

    raw_clock = current_clock
    clock_data[clock_idx] = current_clock
    if clock0 == -1:
        # clock0 = current_clock - period
        clock0 = np.int64(current_clock - period)
        clock0_dec = 0.0

    arg_int = current_clock - clock0  # both int64
    arg = arg_int - clock0_dec
    arg = (arg - period) * 2 * math.pi  # now its a float
    arg = arg / period
    phi0 = math.sin(arg)
    filterr = phi0 + (phi0 - phi_old) * deriv
    freq = 1 / period - filterr * prop

    # this will handle missed clocks
    cycles = round((current_clock - clock0) / period)
    period = 1 / freq
    adj = cycles * period
    adj_int = np.int64(adj)
    adj_dec = adj - adj_int

    clock0 = clock0 + adj_int
    clock0_dec = clock0_dec + adj_dec
    if clock0_dec >= 1:
        int_add = np.int64(clock0_dec)
        clock0 = clock0 + int_add
        clock0_dec = clock0_dec - int_add

    lclock_data[clock_idx] = clock0
    lclock_data_dec[clock_idx] = clock0_dec
    phi_old = phi0
    clock_idx = clock_idx + 1

    return clock0, period, clock_data, clock_idx, phi_old


if __name__ == "__main__":

    print(
        """Custom Measurement example

Implementation of a custom single start, multiple stop measurement, histogramming
the time differences of the two input channels.

The custom implementation will be comparted to a the build-in Histogram class,
which is a multiple start, multiple stop measurement. But for the
selected time span of the histogram, multiple start does not make a difference.
"""
    )
    # fig, ax = plt.subplots()

    ############ depreciated
    # tagger = TimeTagger.createTimeTagger()
    # data_channel = -5
    # clock_channel = 9
    # tagger.setEventDivider(9, 100)
    # tagger.setTriggerLevel(-5, -0.014)
    # tagger.setTriggerLevel(9, 0.05)
    # PLL = CustomPLLHistogram(
    #     tagger,
    #     data_channel,
    #     clock_channel,
    #     10,
    #     phase=0,
    #     deriv=0.001,
    #     prop=2e-10,
    #     n_bins=800000,
    # )
    # # for i in range(40000):
    # #     sleep(.05)
    # #     clocks, pclocks, hist = PLL.getData()
    # #     # print("1: ", clks1[:10])
    # #     # print("HIST: ", hist[:20])
    # #     print("length of clocks: ", len(clocks))
    # #     diff = clocks - pclocks
    # #     print("difference: ", diff[:10])
    # for i in range(40000):
    #     sleep(0.05)
    #     clocks, pclocks, hists = PLL.getData()

    # clocks, pclocks, hists = PLL.getData()

    # basis = np.linspace(clocks[0], clocks[-1], len(clocks))

    buffer = np.zeros((5, 4))
    buffer[0, 0] = 324
    buffer[0, 1] = 345
    buffer[0, 2] = 453
    buffer[0, 3] = 654

    buffer[1, 0] = 654
    buffer[1, 1] = 233
    buffer[1, 2] = 355
    buffer[1, 3] = 767
    print(buffer)
    save_to_buffer(buffer, 1, 4325)

    print(buffer)
