__all__ = ('parse_keyval_string',)

from pyparsing import (
    ParseException,
    FollowedBy,
    Group,
    Suppress,
    Or,
    Optional,
    Word,
    ZeroOrMore,
    alphanums,
    dictOf,
    lineEnd,
    quotedString,
    delimitedList,
    removeQuotes,
)

key = Word(alphanums + '_')
comma = Suppress(',')
raw_t  = Word(alphanums + '@_')
string_t = quotedString.setParseAction(removeQuotes)
single_t = Or([raw_t, string_t])
tuple_t = Suppress('(') + Group(ZeroOrMore(single_t + comma)) + Suppress(')')
value = Or([single_t, tuple_t])
key_value_pair = Group(key + Suppress('=') + value)
key_value_string = ZeroOrMore(key_value_pair + Or([comma, lineEnd]))

def parse_format_header(string):
    '''Generates a variables dictionary from the given string.

    The recognized syntax is as 'key = value' where value could be quoted with
    double or single quotes. Also a value could be a tuple, then it should be
    represented as multiple values which are separated by commas and enclosed
    in bracets. Note that there must be a comma after the last tuple element.
    The whole format string consists of key-value pairs which should be
    separated by commas. "a = 1, b = '1234', c = (9, 8, 7,)" is an example.'''
    try:
        parsed = key_value_string.parseString(string, parseAll = True)
    except ParseException as err:
        raise ValueError('Parsing failed: {}'.format(err)) from None
    return dict(parsed.asList())
