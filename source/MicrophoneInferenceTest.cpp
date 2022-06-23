/*
The MIT License (MIT)

Copyright (c) 2020 EdgeImpulse Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

#include "MicroBit.h"
#include "ContinuousAudioStreamer.h"
#include "StreamNormalizer.h"
#include "Tests.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"

#define INFERENCING_KEYWORD_1     "help"
#define INFERENCING_KEYWORD_2     "arrived"

static NRF52ADCChannel *mic = NULL;
static ContinuousAudioStreamer *streamer = NULL;
static StreamNormalizer *processor = NULL;

static inference_t inference;

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int8_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
    return 0;
}

/**
 * Invoked when we hear the keyword !
 */
static void heard_keyword_1() {
    const char * sad_emoji ="\
        000,255,000,255,000\n\
        000,000,000,000,000\n\
        000,255,255,255,000\n\
        255,000,000,000,255\n\
        000,000,000,000,000\n";
    MicroBitImage img(sad_emoji);
    uBit.io.P0.setDigitalValue(1);
    uBit.display.print(img);
}

static void heard_keyword_2() {
    const char * happy_emoji ="\
        000,255,000,255,000\n\
        000,000,000,000,000\n\
        255,000,000,000,255\n\
        000,255,255,255,000\n\
        000,000,000,000,000\n";
    MicroBitImage img(happy_emoji);
    uBit.io.P0.setDigitalValue(1);
    uBit.display.print(img);
}


/**
 * Invoked when we hear something else
 */
static void heard_other() {
    const char * empty_emoji ="\
        000,000,000,000,000\n\
        000,000,000,000,000\n\
        000,000,255,000,000\n\
        000,000,000,000,000\n\
        000,000,000,000,000\n";
    MicroBitImage img(empty_emoji);
    uBit.display.print(img);
}

void
mic_inference_test()
{
    if (mic == NULL){
        mic = uBit.adc.getChannel(uBit.io.microphone);
        mic->setGain(7,0);          // Uncomment for v1.47.2
        //mic->setGain(7,1);        // Uncomment for v1.46.2
    }

    // alloc inferencing buffers
    inference.buffers[0] = (int8_t *)malloc(EI_CLASSIFIER_SLICE_SIZE * sizeof(int8_t));

    if (inference.buffers[0] == NULL) {
        uBit.serial.printf("Failed to alloc buffer 1\n");
        return;
    }

    inference.buffers[1] = (int8_t *)malloc(EI_CLASSIFIER_SLICE_SIZE * sizeof(int8_t));

    if (inference.buffers[0] == NULL) {
        uBit.serial.printf("Failed to alloc buffer 2\n");
        free(inference.buffers[0]);
        return;
    }

    uBit.serial.printf("Allocated buffers\n");

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = EI_CLASSIFIER_SLICE_SIZE;
    inference.buf_ready = 0;

    mic->output.setBlocking(true);

    if (processor == NULL)
        processor = new StreamNormalizer(mic->output, 0.15f, true, DATASTREAM_FORMAT_8BIT_SIGNED);

    if (streamer == NULL)
        streamer = new ContinuousAudioStreamer(processor->output, &inference);

    uBit.io.runmic.setDigitalValue(1);
    uBit.io.runmic.setHighDrive(true);

    uBit.serial.printf("Allocated everything else\n");

    // number of frames since we heard 'microbit'
    uint8_t last_keywords_1 = 0b0;
    uint8_t last_keywords_2 = 0b0;

    int heard_keyword_1_x_ago = 100;
    int heard_keyword_2_x_ago = 100;

    while(1) {
        uBit.sleep(1);

        if (inference.buf_ready) {
            inference.buf_ready = 0;

            static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

            signal_t signal;
            signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
            signal.get_data = &microphone_audio_signal_get_data;
            ei_impulse_result_t result = { 0 };

            EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, false);
            if (r != EI_IMPULSE_OK) {
                ei_printf("ERR: Failed to run classifier (%d)\n", r);
                return;
            }

            bool heard_keyword_1_this_window = false;
            bool heard_keyword_2_this_window = false;

            if (++print_results >= 0) {
                // print the predictions
                ei_printf("Predictions (DSP: %d ms., Classification: %d ms.): \n",
                    result.timing.dsp, result.timing.classification);
                for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                    ei_printf("    %s: ", result.classification[ix].label);
                    ei_printf_float(result.classification[ix].value);
                    ei_printf("\n");

                    if (strcmp(result.classification[ix].label, INFERENCING_KEYWORD_1) == 0 && result.classification[ix].value > 0.7) {
                        heard_keyword_1_this_window = true;
                    } else if (strcmp(result.classification[ix].label, INFERENCING_KEYWORD_2) == 0 && result.classification[ix].value > 0.15) {
                        heard_keyword_2_this_window = true;
                    }
                    
                }

                last_keywords_1 = last_keywords_1 << 1 & 0x1f;
                last_keywords_2 = last_keywords_2 << 1 & 0x1f;

                if (heard_keyword_1_this_window) {
                    last_keywords_1 += 1;
                } else if (heard_keyword_2_this_window) {
                    last_keywords_2 += 1;
                }

                uint8_t keyword_1_count = 0;
                uint8_t keyword_2_count = 0;
                for (size_t ix = 0; ix < 5; ix++) {
                    keyword_1_count += (last_keywords_1 >> ix) & 0x1;
                    keyword_2_count += (last_keywords_2 >> ix) & 0x1;
                }

                if (heard_keyword_1_this_window) {
                    ei_printf("\nHeard keyword 1: %s (%d times, needs 5)\n", INFERENCING_KEYWORD_1, keyword_1_count);
                } else if (heard_keyword_2_this_window) {
                    ei_printf("\nHeard keyword 2: %s (%d times, needs 5)\n", INFERENCING_KEYWORD_2, keyword_2_count);
                }
                

                if (keyword_1_count >= 1) {
                    ei_printf("\n\n\nDefinitely heard keyword 1: \u001b[32m%s\u001b[0m\n\n\n", INFERENCING_KEYWORD_1);
                    last_keywords_1 = 0;
                    heard_keyword_1_x_ago = 0;
                }
                else {
                    heard_keyword_1_x_ago++;
                }

                if (keyword_2_count >= 1) {
                    ei_printf("\n\n\nDefinitely heard keyword 2: \u001b[32m%s\u001b[0m\n\n\n", INFERENCING_KEYWORD_2);
                    last_keywords_2 = 0;
                    heard_keyword_2_x_ago = 0;
                }
                else {
                    heard_keyword_2_x_ago++;
                }

                if (heard_keyword_1_x_ago <= 4) {
                    heard_keyword_1();
                    ei_printf("LED ON\n");
                } else if (heard_keyword_2_x_ago <= 4) {
                    heard_keyword_2();
                    ei_printf("LED OFF\n");
                } else {
                    heard_other();
                }
                ei_printf("Digital Value %d\n",uBit.io.P0.getDigitalValue());
            }
        }
    }
}


/**
 * Microbit implementations for Edge Impulse target-specific functions
 */
EI_IMPULSE_ERROR ei_sleep(int32_t time_ms) {
    uBit.sleep(time_ms);
    return EI_IMPULSE_OK;
}

void ei_printf(const char *format, ...) {
    char print_buf[1024] = { 0 };

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    uBit.serial.printf("%s", print_buf);
}
