import numpy as np
import xml.etree.ElementTree
import struct
import gzip


def bytes_unzip(data_content, temp_file="/tmp/temp.zip"):
    f = open(temp_file, "wb")
    f.write(data_content)
    f.close()

    with gzip.open(temp_file, 'rb') as f:
        file_content = f.read()

    return file_content


def parse_xml(file_content, header_start_delimiter="^^%\n", header_end_delimiter="^^$$\n"):
    decoded_data = file_content.decode('utf-8', 'ignore')
    sample_start_index = decoded_data.find(header_end_delimiter) + len(header_end_delimiter)

    header = file_content[len(header_start_delimiter):sample_start_index - len(header_end_delimiter)]
    header_namedtuple = parse_header(header)

    data = file_content[sample_start_index:]
    return header_namedtuple, data


def load_raw_waveform(data, data_format, f_s, pad_msb="zeros"):
    datatype = np.dtype(data_format)
    t_s = 1.0 / f_s
    data_length = len(data)

    to_pack = data
    pack_length = data_length
    if pad_msb == "zeros":
        expected_length = int(np.ceil(float(data_length) / datatype.itemsize) * datatype.itemsize)
        padding_length = expected_length - data_length
        if padding_length != 0:
            pack_length = expected_length
            to_pack = data + b"\0" * padding_length

    parsed_values = struct.pack('{}s'.format(pack_length), to_pack)

    data = np.frombuffer(parsed_values, dtype=datatype)
    times = np.arange(0, len(data)) * t_s
    return times, data, f_s


def parse_header(xmlString):
    parsedXml = xml.etree.ElementTree.fromstring(xmlString)
    halo_sample_header = {
        "augurydatVersion": parsedXml.find("augurydatVersion").text,
        #         "bist"=True if parsedXml.find("bist").text else False,
        "clientVersion": parsedXml.find("clientVersion").text,
        "dataFormat": parsedXml.find("dataFormat").text.lower(),
        "endianness": parsedXml.find("endianness").text,
        "Frequency": float(parsedXml.find("Frequency").text),
        "fwVersion": parsedXml.find("fwVersion").text,
        "hwVersion": parsedXml.find("hwVersion").text,
        #         "overflow":True if parsedXml.find("overflow").text else False,
        "Sensor": parsedXml.find("Sensor").text,
        "sensorVersion": parsedXml.find("sensorVersion").text,
        "timestamp": int(float(parsedXml.find("timestamp").text)),
        #         "underrun"=True if parsedXml.find("underrun").text else False,
    }
    return halo_sample_header


def calibrat_data(data, header):
    sensorVersion = header['sensorVersion']
    if sensorVersion == 'HONEYWELL_SS495A':
        factor = 2.5 / 0.003125 / np.sqrt(2)
        adc_resolution_bits = 16
    elif sensorVersion == 'ANALOGDEVICES_ADXL1002-50_1':
        factor = 2.5 / 0.04 / np.sqrt(2)
        adc_resolution_bits = 16
    elif sensorVersion == "ST_IIS3DWB":
        factor = 16 / np.sqrt(2)
        adc_resolution_bits = 16
    else:
        factor = 1.0
        adc_resolution_bits = 1.0
    calibrated_data = data * factor / (2.0 ** (adc_resolution_bits - 1))
    return calibrated_data
