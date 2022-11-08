# -*- coding: utf-8 -*-
# hmm.pxd
# distutils: language=c
#
# Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.

import numpy as np
cimport numpy as cnp
cnp.import_array()


ctypedef cnp.float_t data_t


cdef class HMM:

    cdef list states

    cdef data_t[:] pi
    cdef data_t[:, :] a
    cdef data_t[:] log_pi
    cdef data_t[:, :] log_a
