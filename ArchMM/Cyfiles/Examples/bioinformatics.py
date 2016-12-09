# -*- coding: utf-8 -*-

import theano, sys, pickle, os, random
import numpy as np

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from HMM_Core import *



PRIMARY_SYMBOLS   = {ord(sym) : i for i, sym in enumerate(["A", "R", "N", "D", "C", 
                    "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", 
                    "W", "Y", "V", "B", "Z", "X"])}
SECONDARY_SYMBOLS = {ord(sym) : i for i, sym in enumerate(["C", "H" ,"E", "T"])}
STRUCTURE_SYMBOLS = {"H" : ord("H"), "G" : ord("H"), "I" : ord("H"), "E" : ord("E"),
                    "B" : ord("E"), "T" : ord("T"), "C" : ord("C"), "S" : ord("C"), " " : ord("C")}

class Sequence:
    def __init__(self, data, comments = []):
        """
        Data structure for storing a sequence of amino acids. 
        The latter is represented by a contiguous array of integers.
        The mapping between the amino acids and their numeric value
        is done by using the ascii table.
        
        Attributes
        ----------
        comments [list] : list of informations about the sequence parsed from the FASTA file
                          The list is constructed by splitting the comments using the ' ' delimiter
        N [int] : length of the sequence
        data [np.ndarray] : contiguous array containing the ascii values of the amino acids
        """
        self.comments = comments
        self.N = len(data)
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            # If a string is passed, the latter is converted to a numpy array
            self.data = np.empty(self.N, dtype = np.int16)
            for i in range(self.N):
                self.data[i] = Sequence.charToInt(data[i])
    @staticmethod
    def charToInt(c):
        return ord(c)
    @staticmethod
    def intToChar(c):
        return chr(c)
    def computeIdentity(self, other):
        """ The identity rate between the two sequences (self and other) 
        is determined by the number of a.a. pairs that are identical 
        and the total number of a.a. pairs 
        The function assumes that the two sequences have the same length. """
        assert(self.N == other.N)
        return (self.N - np.count_nonzero(self.data[:] - other.data[:])) / self.N
    def getArray(self, convert_to_indexes = False, symbol_dict = PRIMARY_SYMBOLS):
        if convert_to_indexes:
            data = np.empty(len(self.data), dtype = self.data.dtype)
            for i in range(len(self.data)):
                try:
                    data[i] = symbol_dict[self.data[i]]
                except KeyError:
                    data[i] = np.random.randint(0, len(symbol_dict))
            return data
        else:
            return self.data
    def __len__(self):
        return self.N
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Created a new sequence containing the requested subset
            return Sequence(self.data[key], self.comments)
        else:
            return self.data[key]
    def __setitem__(self, key, value):
        if isinstance(value, Sequence):
            # Assigning multiple values at once
            self.data[key] = value.data[:]
        else:
            self.data[key] = value
    def __str__(self):
        """ Converts the a.a. sequence back to its string representation """
        s = ""
        for i in range(self.N):
            s += Sequence.intToChar(self.data[i])
        return s
    def __repr__(self):
        """ Returns the information/comments about the sequence """
        s = ""
        for comment in self.comments:
            s += comment.rstrip() + " "
        return s
    def __eq__(self, other):
        """ Checks if two sequences are perfectly identical """
        return (self.N == other.N) and (np.count_nonzero(self.data[:] - other.data[:]) == self.N)
    
    
class Parser:
    MIN_DATA_PER_LINE = 10
    
    def __init__(self, dataset_folder, info_filepath):
        self.dataset_folder = dataset_folder
        self.info_filepath = info_filepath
        self.data = None
        
    @staticmethod
    def charToInt(c):
        if c.islower():
            return ord(c.upper())
        return ord(c)
        
    def parse(self):
        """
        try:
            os.chmod(self.dataset_folder, 777)
        except OSError:
            pass
        """
        self.data = list()
        info_file = open(self.info_filepath, "r")
        for line in info_file.readlines():
            line = line.replace('\n', '')
            elements = line.split(" ")
            assert(len(elements) == 2)
            sequence_name, annotations = elements[0], elements[1]
            assert(len(sequence_name) == 5)
            filename = "%s.dssp" % sequence_name[:4]
            self.data.append(self.parseDSSP(filename, sequence_name[-1]))
        return self.data
    
    @staticmethod
    def parseLineFromHeader(line):
        return line.split(':')[1].replace(';', '').split("  ")[0]
                
    def parseDSSP(self, filename, sequence_id):
        try:
            with open(os.path.join(self.dataset_folder, filename)) as dssp_file:
                data = list()
                line = dssp_file.readline()
                aa_locations = []
                while line:
                    line_split = line.split()
                    if len(line) > Parser.MIN_DATA_PER_LINE and line_split[0] == "COMPND":
                        compnd  = Parser.parseLineFromHeader(line)
                        line = dssp_file.readline()
                        organism_name = Parser.parseLineFromHeader(line)
                        break
                    line = dssp_file.readline()
                while line:
                    line_split = line.split()
                    if len(line) > Parser.MIN_DATA_PER_LINE and line_split[0] == "#":
                        break
                    line = dssp_file.readline()
                residue_index = line.find("RESIDUE") + len("RESIDUE") - 1
                aa_index = line.find("AA")
                structure_index = line.find("STRUCTURE")
                line = dssp_file.readline()
                while line:
                    if len(line) > Parser.MIN_DATA_PER_LINE:
                        if line[residue_index] == sequence_id:
                            line_data = [Parser.charToInt(line[residue_index]), 
                                         Parser.charToInt(line[aa_index]), 
                                         STRUCTURE_SYMBOLS[line[structure_index]]]
                            aa_locations.append(int(line[:residue_index-1].split()[1]))
                            data.append(np.array(line_data, dtype = np.int16))

                    line = dssp_file.readline()
            data = np.array(data, dtype = np.int16)
            start, end = min(aa_locations), max(aa_locations)
            return [data, compnd, organism_name, filename, slice(start, end, 1)]
        except FileNotFoundError:
            print("Warning : file %s could not be found" % filename)
            
    def convertToArrays(self):
        sequences = list()
        for s in self.data:
            relevant_columns, compnd, organism_name = s[0], s[1], s[2]
            protein_sequence = Sequence(relevant_columns[:, 1])
            protein_sequence = protein_sequence.getArray(
                convert_to_indexes = True, symbol_dict = PRIMARY_SYMBOLS)
            secondary_structure = Sequence(relevant_columns[:, 2])
            secondary_structure = secondary_structure.getArray(
                convert_to_indexes = True, symbol_dict = SECONDARY_SYMBOLS)
            sequences.append((protein_sequence, secondary_structure))
        return sequences
            
    def writeFASTA(self, filename):
        with open(filename, "w") as output_file:
            for s in self.data:
                relevant_columns, compnd, organism_name = s[0], s[1], s[2]
                identifier = s[3].split('.')[0]
                start, end = s[4].start, s[4].stop
                protein_sequence = Sequence(relevant_columns[:, 1])
                secondary_structure = Sequence(relevant_columns[:, 2])
                output_file.write("> %s|%s|%i-%i /%s\n" % (identifier, compnd, start, end, organism_name))
                output_file.write("%s\n" % str(protein_sequence))
                output_file.write("%s\n" % str(secondary_structure))
                
    def parseFASTA(self, filename):
        """ Parses a FASTA file and returns a list of Sequence objects """
        sequences = []
        f = open(filename, 'r')
        seq = ""
        comments = []
        for line in f:
            if line[0] == '>': # If line contains comments/information
                line = line.split(' ')
                if len(line) > 0: # Ensure that the comments' line is not empty
                    line[0] = line[0][1:]
                if len(seq) > 0: # If the a.a. chars have been parsed
                    sequences.append(Sequence(seq, comments))
                    seq = ""
                comments = line
            else:
                seq += line.rstrip()
        sequences.append(Sequence(seq, comments))
        f.close()
        return sequences

config = IOConfig()
config.n_classifiers = 1
config.n_states = 5
config.architecture = "linear"
config.n_iterations = 50
config.pi_learning_rate = 0.005
config.pi_nhidden = 5
config.pi_nepochs = 2
config.pi_activation = "sigmoid"
config.s_learning_rate  = 0.005
config.s_nhidden  = 5
config.s_nepochs = 2
config.s_activation = "sigmoid"
config.o_learning_rate  = 0.005
config.o_nhidden  = 5
config.o_nepochs = 2
config.o_activation = "sigmoid"
config.missing_value_sym = np.nan

if __name__ == "__main__":
    parser = Parser("MP4_data_2016/dataset/dssp", "MP4_data_2016/dataset/CATH_info.txt")
    parser.parse()
    training_sequences = parser.convertToArrays()
    X_traning, y_training = list(), list()
    k = 0
    for (primary, secondary) in training_sequences:
        print(primary)
        s = np.concatenate((
            np.random.randint(0, 22, size = 8), 
            primary,    
            np.random.randint(0, 22, size = 8)))
        X = np.zeros((len(primary), 17 * 24), dtype = np.float32)
        for i in range(8, len(primary) + 8):
            for d in range(-8, 9):
                X[i - 8, (d + 8) * 17 + primary[i - 8]] = 1
        X_traning.append(X)
        y_training.append(y_training)
        k += 1
        if k == 2:
            break
    pickle.dump((X_traning, y_training), open("temp", "wb"))
    print(X_traning[0].shape)

    iohmm = AdaptiveHMM(config.n_states, config.architecture, has_io = True)
    fit = iohmm.fit(X_traning, targets = y_training, n_classes = 24,
                is_classifier = True, parameters = config)
    iohmm.pySave(os.path.join("classifier"))
