#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Read VIIRS data from framed raw CADU data (RS decoded).

At the moment, only reading single gain predictor bands M12, M10, M9, M6, M16, M15 works.

Mostly from

Joint Polar Satellite System (JPSS)
Common Data Format Control Book –
External
Volume VII – Part I
JPSS Downlink Data Formats


TODO:
- implement support for DNB and I bands
- support for reed-solomon decoding
- use pyorbital to compute geolocation data
- calibrate the channels to radiances and/or bt/reflectances

"""


import sys
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

import risotto as rice
from trollimage.image import Image

#from reed import rs
# from zfec import Decoder
# from trollcast.formats.reedsolomon import *
#from reedsolo import RSCodec

table = np.array([0xff, 0x48, 0x0e, 0xc0, 0x9a, 0x0d, 0x70, 0xbc, 0x8e, 0x2c,
                  0x93, 0xad, 0xa7, 0xb7, 0x46, 0xce, 0x5a, 0x97, 0x7d, 0xcc,
                  0x32, 0xa2, 0xbf, 0x3e, 0x0a, 0x10, 0xf1, 0x88, 0x94, 0xcd,
                  0xea, 0xb1, 0xfe, 0x90, 0x1d, 0x81, 0x34, 0x1a, 0xe1, 0x79,
                  0x1c, 0x59, 0x27, 0x5b, 0x4f, 0x6e, 0x8d, 0x9c, 0xb5, 0x2e,
                  0xfb, 0x98, 0x65, 0x45, 0x7e, 0x7c, 0x14, 0x21, 0xe3, 0x11,
                  0x29, 0x9b, 0xd5, 0x63, 0xfd, 0x20, 0x3b, 0x02, 0x68, 0x35,
                  0xc2, 0xf2, 0x38, 0xb2, 0x4e, 0xb6, 0x9e, 0xdd, 0x1b, 0x39,
                  0x6a, 0x5d, 0xf7, 0x30, 0xca, 0x8a, 0xfc, 0xf8, 0x28, 0x43,
                  0xc6, 0x22, 0x53, 0x37, 0xaa, 0xc7, 0xfa, 0x40, 0x76, 0x04,
                  0xd0, 0x6b, 0x85, 0xe4, 0x71, 0x64, 0x9d, 0x6d, 0x3d, 0xba,
                  0x36, 0x72, 0xd4, 0xbb, 0xee, 0x61, 0x95, 0x15, 0xf9, 0xf0,
                  0x50, 0x87, 0x8c, 0x44, 0xa6, 0x6f, 0x55, 0x8f, 0xf4, 0x80,
                  0xec, 0x09, 0xa0, 0xd7, 0x0b, 0xc8, 0xe2, 0xc9, 0x3a, 0xda,
                  0x7b, 0x74, 0x6c, 0xe5, 0xa9, 0x77, 0xdc, 0xc3, 0x2a, 0x2b,
                  0xf3, 0xe0, 0xa1, 0x0f, 0x18, 0x89, 0x4c, 0xde, 0xab, 0x1f,
                  0xe9, 0x01, 0xd8, 0x13, 0x41, 0xae, 0x17, 0x91, 0xc5, 0x92,
                  0x75, 0xb4, 0xf6, 0xe8, 0xd9, 0xcb, 0x52, 0xef, 0xb9, 0x86,
                  0x54, 0x57, 0xe7, 0xc1, 0x42, 0x1e, 0x31, 0x12, 0x99, 0xbd,
                  0x56, 0x3f, 0xd2, 0x03, 0xb0, 0x26, 0x83, 0x5c, 0x2f, 0x23,
                  0x8b, 0x24, 0xeb, 0x69, 0xed, 0xd1, 0xb3, 0x96, 0xa5, 0xdf,
                  0x73, 0x0c, 0xa8, 0xaf, 0xcf, 0x82, 0x84, 0x3c, 0x62, 0x25,
                  0x33, 0x7a, 0xac, 0x7f, 0xa4, 0x07, 0x60, 0x4d, 0x06, 0xb8,
                  0x5e, 0x47, 0x16, 0x49, 0xd6, 0xd3, 0xdb, 0xa3, 0x67, 0x2d,
                  0x4b, 0xbe, 0xe6, 0x19, 0x51, 0x5f, 0x9f, 0x05, 0x08, 0x78,
                  0xc4, 0x4a, 0x66, 0xf5, 0x58], dtype=np.uint8)


# Rice decoding parameters
J = 8
refblk = 128 * J


def show(data, filename=None, stripes=0):
    """Show the stetched data.
    """

    norm_arr = np.array((data - data.min()) * 255.0 /
                        (data.max() - data.min())).astype(np.uint8)
    if stripes:
        r_norm_arr = np.array(norm_arr)
        r_norm_arr[:, ::stripes] = np.uint8(255 - r_norm_arr[:, ::stripes])
        img = Image((r_norm_arr, norm_arr, norm_arr), mode="RGB")
        img.stretch("crude")
    else:
        from PIL import Image as pil
        img = pil.fromarray(norm_arr)

    # img = Image((ch14, ch12, ch09), mode="RGB")
    # img = Image((ch14[:, 38:], ch12[:, 38:], ch09[:, 38:]), mode="RGB")

    if filename:
        img.save(filename)
    else:
        img.show()


#############################################################

vcdu_type = np.dtype([('version', '>u2'),
                      ('count', '>u1', (3,)),
                      ('replay', '>u1')])

ccsds_phdr = np.dtype([('ccsds_version', '>u2'),
                       ('sequence', '>u2'),
                       ('packet_length', '>u2')])

ccsds_shdr = np.dtype([('day', '>u2'),
                       ('milliseconds', '>u4'),
                       ('microseconds', '>u2'),
                       ('seq_count', '>u1'),
                       ('spare', '>u1')])


def decode_str(arr):
    """do pseudo-noise decoding.
    """
    decoder = np.tile(table, np.ceil(len(arr) / 255.0))
    return (np.fromstring(arr, np.uint8) ^ decoder[:len(arr)]).tostring()


def cadu_info(line):
    print "=" * 3, "Cadu Info", "=" * 3
    print "version", (line["version"] >> 13) & 7
    print "type", (line["version"] >> 12) & 1
    print "sec hdr flag", (line["version"] >> 11) & 1
    print "apid", bin((line["version"]) & 2047)

    print "sequence flag", (line["sequence"] >> 14) & 3
    print "sequence count", (line["sequence"]) & 0x3fff
    print "packet length", (line["plen"])

    print "timestamp", line["days"], line["milliseconds"]


def read_vcdu_hdr(packets):
    """split *packets* into header and body.
    """
    for packet in packets:
        yield np.fromstring(packet, vcdu_type, count=1)[0], packet[6:]


def reed_decode(packets):
    #codec = rs.RSCoder(255, 223)
    res = bytearray("\0" * 892)
    # codec = RSCodec(32)

    for packet in packets:
        # do the decoding!
        yield packet[:-128]
        # for i in range(4):
        #     #print i
        #     try:
        #         print len(packet[i::4])
        #         print codec.verify(packet[i::4])
        #         decoded = codec.decode(packet[i::4])
        #         if not decoded:
        #             raise ValueError
        #         #print decoded
        #         res[i::4] = decoded
        #         #print "decoded"
        #         #raw_input()
        #     except IOError:
        #         print "uncorrectable"
        #         res[i::4] = packet[i:892:4]
        #     except ReedSolomonError as err:
        #         #print "uncorrectable:", err
        #         res[i::4] = packet[i:892:4]
        # print str(res[:892]) == packet[:892]
        # yield str(res[:892])


def pn_decode(packets, dlen=0, skip=False):
    """Do pseudo-noise decoding of the *packets* if *skip* isn't set to True.
    """
    for packet in packets:
        if skip:
            yield packet
        else:
            yield packet[:dlen] + decode_str(packet[dlen:])


def frame_sync(packets):
    """Phony frame_sync.
    """
    for packet in packets:
        yield packet[4:]


def file_reader(filename, plen=1024):
    """Iterate of the file designated with *filename*.
    """
    with open(filename) as fd:
        while True:
            packet = fd.read(plen)
            if packet:
                yield packet
            else:
                return


def get_vc(packets, req_vcid):
    """Filter out *packets* not matching *req_vcid*.
    """
    cnt = 0
    for hdr, packet in packets:
        vcid = hdr["version"] & (2 ** 6 - 1)
        if vcid == req_vcid:
            yield hdr, packet
            cnt += 1
            if cnt % 10000 == 0:
                # return
                print "vcdu packets read", cnt


def ccsds_packets(vcdu_packets):
    """Assemble *vcdu_packets* into ccsds packets
    """
    prev_count = None
    current_buffer = ""
    chdr = None
    missing_packets = 0
    total_packets = 0
    for hdr, line in vcdu_packets:
        count = np.sum(hdr["count"] << (8 * np.array([2, 1, 0])))
        if prev_count is not None and (count - prev_count) % 2**24 != 1:
            current_buffer = ""
            prev_count = count
            missing_packets += ((count - prev_count) % 2**24) - 1
            continue
        prev_count = count
        pointer = np.fromstring(line, ">u2", count=1)[0] & 0x7ff
        if pointer == 0x7ff:
            if current_buffer == "":
                missing_packets += 1
                continue
            else:
                current_buffer += line[2:]
        else:
            if current_buffer != "":
                current_buffer += line[2:pointer + 2]
                total_packets += 1
                yield chdr, current_buffer
                current_buffer = ""
            if len(line[pointer + 2:]) < ccsds_phdr.itemsize:
                current_buffer = line[pointer + 2:]
                continue

            offset = pointer + 2

            chdr = np.fromstring(line[offset:], ccsds_phdr, count=1)[0]
            plen = chdr["packet_length"] + 7

            while plen <= len(line[offset:]):
                total_packets += 1
                yield chdr, line[offset:offset + plen]
                current_buffer = ""
                offset += plen
                if len(line[offset:]) < ccsds_phdr.itemsize:
                    break
                chdr = np.fromstring(line[offset:], ccsds_phdr, count=1)[0]
                plen = chdr["packet_length"] + 7

            current_buffer = line[offset:]
    print missing_packets * 1.0 / total_packets


def get_ap(packets, req_apid):
    """Filter out *packets* not matching *req_apid*.
    """
    cnt = 0
    for hdr, packet in packets:
        apid = hdr["ccsds_version"] & (2**11 - 1)
        if apid == req_apid:
            yield(hdr, packet)
            cnt += 1
            if cnt % 10000 == 0:
                print "ccsds packets read", cnt


def ccsds_info(hdr):
    version = hdr["ccsds_version"] >> 13
    type_ind = bool((hdr["ccsds_version"] >> 12) & 1)
    sec_hdr = bool((hdr["ccsds_version"] >> 11) & 1)
    apid = hdr["ccsds_version"] & (2**11 - 1)

    seq_flag = (hdr["sequence"] >> 14) & 3
    seq_count = hdr["sequence"] & 0x3fff

    plen = hdr["packet_length"]

    print version, type_ind, sec_hdr, apid
    print seq_flag, seq_count
    print plen


def data_packets(packets):
    """Gather *packets* into logical bunches (start, middle and end packets together).
    """
    current_buffer = {}
    for hdr, packet in packets:
        sec_hdr = bool((hdr["ccsds_version"] >> 11) & 1)
        seq_flag = (hdr["sequence"] >> 14) & 3
        apid = hdr["ccsds_version"] & (2**11 - 1)
        # print len(packet) - hdr["packet_length"]
        # if (len(packet) - 7) != hdr["packet_length"]:
        #    print "skip"
        #    continue
        if seq_flag == 0b11:
            # standalone packet
            # print "standalone"
            yield apid, [packet[6:]]
            current_buffer[apid] = []
        elif seq_flag == 0b01:
            # start packet
            current_buffer[apid] = [packet[6:]]
        elif seq_flag == 0b00:
            # continuation packet
            if len(current_buffer[apid]) == 0:
                current_buffer[apid].append(None)
                print "missing start packet"
            current_buffer[apid].append(packet[6:])
        elif seq_flag == 0b10:
            # end packet
            current_buffer[apid].append(packet[6:])
            yield apid, current_buffer[apid]
            current_buffer[apid] = []

middle_hdr = np.dtype([('viirs_sequence_count', '>u4'),
                       ('packet_time', '>u8'),
                       ('format_version', '>u1'),
                       ('instrument_number', '>u1'),
                       ('spare', '>u2')])
detector_hdr = np.dtype([('integrity', '>u2'),
                         ('band', '>u1'),
                         ('detector', '>u1'),
                         ('sync_word_pattern', '>u4'),
                         ('reserved', '>u1', (64,))])
detector_bef = np.dtype([('fill_data', '>u2'),
                         ('checksum_offset', '>u2')])
detector_aft = np.dtype([('checksum', '>u4'),
                         ('sync_word', '>u4')])


class SyncError(Exception):
    pass


def decode_line(middle_pack):
    """Decode the scanline using the rice decoder.
    """
    det_hdr = np.fromstring(middle_pack[middle_hdr.itemsize:],
                            dtype=detector_hdr, count=1)[0]
    sync_word_pattern = det_hdr['sync_word_pattern']
    # print "detector", det_hdr["detector"], len(middle_pack)
    offset = middle_hdr.itemsize + detector_hdr.itemsize
    # aggr 1
    data_len = np.fromstring(middle_pack[offset:],
                             dtype=detector_bef, count=1)["checksum_offset"][0] - 4
    offset += detector_bef.itemsize
    # data1 = np.fromstring(
    #    middle_pack[offset:offset + data_len], dtype=np.uint8)
    data1 = bytearray(middle_pack[offset:offset + data_len])
    offset += data_len
    det_aft = np.fromstring(middle_pack[offset:],
                            dtype=detector_aft, count=1)[0]
    if not det_aft["sync_word"] == sync_word_pattern:
        raise SyncError
    offset += detector_aft.itemsize
    # aggr 2
    data_len = np.fromstring(middle_pack[offset:],
                             dtype=detector_bef, count=1)["checksum_offset"][0] - 4
    offset += detector_bef.itemsize
    # data2 = np.fromstring(
    #    middle_pack[offset:offset + data_len], dtype=np.uint8)
    data2 = bytearray(middle_pack[offset:offset + data_len])
    offset += data_len
    det_aft = np.fromstring(middle_pack[offset:],
                            dtype=detector_aft, count=1)[0]
    if not det_aft["sync_word"] == sync_word_pattern:
        raise SyncError
    offset += detector_aft.itemsize
    # aggr 3
    det_bef = np.fromstring(middle_pack[offset:],
                            dtype=detector_bef, count=1)[0]
    data_len = np.fromstring(middle_pack[offset:],
                             dtype=detector_bef, count=1)["checksum_offset"][0] - 4
    offset += detector_bef.itemsize

    # data3 = np.fromstring(
    #    middle_pack[offset:offset + data_len], dtype=np.uint8)
    data3 = bytearray(middle_pack[offset:offset + data_len])
    offset += data_len
    det_aft = np.fromstring(middle_pack[offset:],
                            dtype=detector_aft, count=1)[0]
    if not det_aft["sync_word"] == sync_word_pattern:
        raise SyncError
    offset += detector_aft.itemsize
    # print bin(det_bef["fill_data"]), data_len,
    # aggr 4
    data_len = np.fromstring(middle_pack[offset:],
                             dtype=detector_bef, count=1)["checksum_offset"][0] - 4
    offset += detector_bef.itemsize
    # data4 = np.fromstring(
    #    middle_pack[offset:offset + data_len], dtype=np.uint8)
    data4 = bytearray(middle_pack[offset:offset + data_len])
    offset += data_len
    det_aft = np.fromstring(middle_pack[offset:],
                            dtype=detector_aft, count=1)[0]
    if not det_aft["sync_word"] == sync_word_pattern:
        raise SyncError
    offset += detector_aft.itemsize
    # aggr 5
    data_len = np.fromstring(middle_pack[offset:],
                             dtype=detector_bef, count=1)["checksum_offset"][0] - 4
    offset += detector_bef.itemsize
    # data5 = np.fromstring(
    #    middle_pack[offset:offset + data_len], dtype=np.uint8)
    data5 = bytearray(middle_pack[offset:offset + data_len])
    offset += data_len
    det_aft = np.fromstring(middle_pack[offset:],
                            dtype=detector_aft, count=1)[0]
    if not det_aft["sync_word"] == sync_word_pattern:
        raise SyncError
    offset += detector_aft.itemsize
    # aggr 6
    data_len = np.fromstring(middle_pack[offset:],
                             dtype=detector_bef,
                             count=1)["checksum_offset"][0] - 4
    offset += detector_bef.itemsize
    # data6 = np.fromstring(
    #    middle_pack[offset:offset + data_len], dtype=np.uint8)
    data6 = bytearray(middle_pack[offset:offset + data_len])
    offset += data_len
    det_aft = np.fromstring(middle_pack[offset:],
                            dtype=detector_aft, count=1)[0]
    if not det_aft["sync_word"] == sync_word_pattern:
        raise SyncError
    offset += detector_aft.itemsize

    return (data1, data2, data3, data4, data5, data6)


def read6(filename, vcid, apids, pn=False):
    """Read the data from *filename* for a given *vcid* and *apid*."""
    return data_packets(
        get_aps(
            ccsds_packets(
                get_vc(
                    read_vcdu_hdr(
                        reed_decode(
                            pn_decode(
                                frame_sync(
                                    file_reader(filename)),
                                skip=not pn))),
                    vcid)),
            apids))


def ccsds_iterator(filename, vcid, pn=False):
    return ccsds_packets(
        get_vc(
            read_vcdu_hdr(
                reed_decode(
                    pn_decode(
                        frame_sync(
                            file_reader(filename)),
                        skip=not pn))),
            vcid))

apids = {800: "M04",
         801: "M05",
         802: "M03",
         803: "M02",
         804: "M01",
         805: "M06",
         806: "M07",
         807: "M09",
         808: "M10",
         809: "M08",
         810: "M11",
         811: "M13",
         812: "M12",
         813: "I04",
         814: "M16",
         815: "M15",
         816: "M14",
         817: "I05",
         818: "I01",
         819: "I02",
         820: "I03",
         821: "DNB",
         822: "DNBMGS",
         823: "DNBLGS",
         825: "Calibration",
         826: "Engineering"}

channels = {v: k for k, v in apids.items()}

dual = [800, 801, 802, 803, 804, 806, 811]
single = [805, 807, 808, 809, 810, 812, 814, 815, 816]


def build_scan(bunch, apid):
    """Convert the packet *bunch* into a scan.
    """
    if len(bunch) == 1:
        return
    print "len(bunch)", len(bunch), apid
    scan = []
    last_det = 0

    if apid in single:
        agg_zone0 = 640
        agg_zone1 = 368
        agg_zone2 = 592
    elif apid in dual:
        agg_zone0 = 640
        agg_zone1 = 368 * 2
        agg_zone2 = 592 * 3
    else:
        raise ValueError

    for middle_pack in bunch[1:]:

        det_hdr = np.fromstring(middle_pack[middle_hdr.itemsize:],
                                dtype=detector_hdr, count=1)[0]
        # print det_hdr["band"], det_hdr["detector"]
        if last_det + 1 != det_hdr["detector"]:
            for i in range(det_hdr["detector"] - (last_det + 1)):
                scan.append(np.zeros(agg_zone0 * 2 +
                                     agg_zone1 * 2 + agg_zone2 * 2))
        last_det = det_hdr["detector"]
        try:
            d1, d2, d3, d4, d5, d6 = decode_line(middle_pack)
        except (SyncError, ValueError):
            print "skipping"
            scan.append(np.zeros(agg_zone0 * 2 + agg_zone1 * 2 + agg_zone2 * 2))
            continue

        if len(d1) != 4:
            d1 = np.concatenate([np.array(chunk)
                                 for chunk in rice.decode(d1, 15, J, refblk)])
            if len(d1) != agg_zone0:
                d1 = np.zeros(agg_zone0, dtype=np.int64)
        else:
            d1 = np.zeros(agg_zone0, dtype=np.int64)

        if len(d6) != 4:
            d6 = np.concatenate([np.array(chunk)
                                 for chunk in rice.decode(d6, 15, J, refblk)])
            if len(d6) != agg_zone0:
                d6 = np.zeros(agg_zone0, dtype=np.int64)
        else:
            d6 = np.zeros(agg_zone0, dtype=np.int64)

        if len(d2) != 4:
            d2 = np.concatenate([np.array(chunk)
                                 for chunk in rice.decode(d2, 15, J, refblk)])
            if len(d2) != agg_zone1:
                print 'got', len(d2)
                d2 = np.zeros(agg_zone1, dtype=np.int64)
        else:
            d2 = np.zeros(agg_zone1, dtype=np.int64)

        if len(d5) != 4:
            d5 = np.concatenate([np.array(chunk)
                                 for chunk in rice.decode(d5, 15, J, refblk)])
            if len(d5) != agg_zone1:
                d5 = np.zeros(agg_zone1, dtype=np.int64)
        else:
            d5 = np.zeros(agg_zone1, dtype=np.int64)

        d3 = np.concatenate([np.array(chunk)
                             for chunk in rice.decode(d3, 15, J, refblk)])
        if len(d3) != agg_zone2:
            d3 = np.zeros(agg_zone2, dtype=np.int64)
        d4 = np.concatenate([np.array(chunk)
                             for chunk in rice.decode(d4, 15, J, refblk)])
        if len(d4) != agg_zone2:
            d4 = np.zeros(agg_zone2, dtype=np.int64)
        scan.append(np.hstack((d1, d2, d3, d4, d5, d6)))

    if last_det < 15:
        for i in range(15 - last_det):
            scan.append(np.zeros(agg_zone0 * 2 + agg_zone1 * 2 + agg_zone2 * 2))
    return apid, np.vstack(reversed(scan))


def get_aps(packets, req_apids):
    """Filter out *packets* not matching *req_apid*.
    """
    cnt = 0
    for hdr, packet in packets:
        apid = hdr["ccsds_version"] & (2**11 - 1)
        if apid in req_apids:
            yield(hdr, packet)
            cnt += 1
            if cnt % 10000 == 0:
                print "ccsds packets read", cnt

dependencies = {'M05': 'M04',
                'M03': 'M04',
                'M02': 'M03',
                'M01': 'M02',
                'M08': 'M10',
                'M11': 'M10',
                'I04': 'M12',
                'M14': 'M15',
                'I05': 'M15',
                'I02': 'I01',
                'I03': 'I02',
                'M07': 'I04'
                }


def get_deps(channel):
    res = [channel]
    try:
        res.extend(get_deps(dependencies[channel]))
    except KeyError:
        pass
    return res

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cadu_file", help="cadu file to read")
    parser.add_argument("channel", nargs='+', help="channel to read")
    parser.add_argument("-p", '--pn-decode',
                        action='store_true', help="perform PN decoding")
    opts = parser.parse_args()

    packs = []
    test = []
    zone = []
    #chs = read5(sys.argv[1])
    #channels = sort_ap(ccsds_packets(chs[16]))
    # for pack in data_packets(channels[812]):

    now = datetime.now()

    # M12 812 ok.
    # M10 808 ok.
    # M9  807 ok.
    # M6  805 nok. Wrapping ?
    # M16 814 ok.
    # M15 815 ok.

    try:
        reqs = opts.channel
        to_read = []
        for req in opts.channel:
            to_read.extend(get_deps(req))
        print 'reading', to_read
        apids_to_read = [channels[channel] for channel in to_read]
        apids_to_read.reverse()
        zone = Parallel(n_jobs=4)(delayed(build_scan)(pack, apid)
                                  for apid, pack in read6(opts.cadu_file,
                                                          vcid=16,
                                                          apids=apids_to_read,
                                                          pn=opts.pn_decode)
                                  if len(pack) > 1)
        # zone = [build_scan(pack, apid)
        # for apid, pack in read6(sys.argv[1], vcid=16, apids=to_read) if
        # len(pack) > 1]

        print "reading took", datetime.now() - now

        scans = {}
        the_data = {}
        for apid, scan in zone:
            scans.setdefault(apid, []).append(scan)
        for apid in apids_to_read:
            zone = scans[apid]
            the_data[apid] = np.vstack(zone)
            the_data[apid] = the_data[apid].astype(np.int64) & (2 ** 13 - 1)
            try:
                depapid = channels[dependencies[apids[apid]]]
                the_data[apid] += the_data[depapid] - (2**14 - 1)
            except KeyError:
                pass
            # show(the_data)
            img = Image([the_data[apid].astype(np.float32)], mode="L")
            img.stretch("histogram")
            img.show()
            import os
            fname = os.path.splitext(os.path.basename(sys.argv[1]))[
                0] + apids[apid] + '.png'
            print 'saving', fname
            img.save(fname)

        #img =  Image([the_data[801], the_data[800], the_data[802]], mode="RGB")
        # img.stretch("histogram")
        # img.show()
    except Exception as err:
        raise
