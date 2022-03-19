import csv
from datetime import datetime, timedelta, date
import argparse
import os
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
from run_ga import run_ga


class Person:
    def __init__(self, id, name, max_hours_per_week, max_hours_per_day, start_time, end_time, roles):
        self.id = id
        self.name = name
        self.max_hours_per_week = int(max_hours_per_week)
        self.max_hours_per_day = int(max_hours_per_day)
        self.start_time = start_time
        self.end_time = end_time
        self.roles = roles

    @staticmethod
    def from_csv_dict(id, csv_dict):
        return Person(id, **csv_dict)


def hour_string_to_hours(hour_string):
    def _(h):
        return int(h.replace(':00', ''))
    start, end = hour_string.split('-')
    return set(range(_(start), _(end) + 1))


def day_string_to_days(day_string):
    start, end = day_string.split('-')
    days = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    return set(range(days[start], days[end] + 1))


def datetime_in_range(datetime, hour_string, day_string):
    valid_hours = hour_string_to_hours(hour_string)
    valid_days = day_string_to_days(day_string)
    return datetime.weekday() in valid_days and datetime.hour in valid_hours


class Constraint:
    def is_satisfied_by_solution(self, solution):
        raise NotImplementedError(self)


class HourAndDayRangeConstraint(Constraint):
    def __init__(self, hour_string, day_string):
        self.hour_string = hour_string
        self.day_string = day_string

    def hours_to_schedule_in_range(self, start_date, end_date):
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.min.time())
        hours = set()
        current_date = start_date
        while current_date < end_date:
            if datetime_in_range(current_date, self.hour_string, self.day_string):
                hours.add(current_date)
            current_date += timedelta(hours=1)
        return hours


class ScheduledHoursConstraint(HourAndDayRangeConstraint):
    value = 100

    def is_satisfied_by_solution(self, solution):
        for hour in self.hours_to_schedule_in_range(solution.from_date, solution.to_date):
            if hour not in solution.allocations:
                return False
        return True


class RequiredRoleConstraint(HourAndDayRangeConstraint):
    value = 100
    def __init__(self, hour_string, day_string, role):
        super().__init__(hour_string, day_string)
        self.role = role

    def is_satisfied_by_solution(self, solution):
        for hour in self.hours_to_schedule_in_range(solution.from_date, solution.to_date):
            if hour not in solution.allocations:
                return False
            for person in solution.people_scheduled_for_hour(hour):
                if self.role in person.roles:
                    break
            else:
                 return False   
        return True


class MaxHoursPerWeekConstraint(Constraint):
    value = 10

    def __init__(self, person):
        self.person = person

    def is_satisfied_by_solution(self, solution):
        hours = 0
        for allocation in solution.allocations.values():
            if self.person.id == allocation:
                hours += 1
        return hours <= self.person.max_hours_per_week


class MaxHoursPerDayConstraint(Constraint):
    value = 10

    def __init__(self, person):
        self.person = person

    def is_satisfied_by_solution(self, solution):
        hours = 0
        for allocation in solution.allocations.values():
            if self.person.id == allocation:
                hours += 1
        return hours <= self.person.max_hours_per_day


constraint_builders = {
    'scheduled-hours': ScheduledHoursConstraint,
    'required-role': RequiredRoleConstraint
}


def import_people(csv_file):
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        return [Person.from_csv_dict(i, n) for i, n in enumerate(reader)]


def import_constraints(csv_file):
    with open(csv_file) as f:
        reader = csv.reader(f)
        return [constraint_builders[constraint_type](*args)
                for constraint_type, *args in reader]


def get_parser():
    parser = argparse.ArgumentParser(description='Run GA test')
    parser.add_argument('--from', type=date.fromisoformat, help='ISO format, inclusive', required=True)
    parser.add_argument('--to', type=date.fromisoformat, help='ISO format, exclusive', required=True)
    parser.add_argument('--test', type=str, help='Test directory name', required=True)
    parser.add_argument('--num-generations', required=False, default=100, type=int)
    parser.add_argument('--num-parents-mating', required=False, default=20, type=int)
    parser.add_argument('--sol-per-pop', required=False, default=100, type=int)
    parser.add_argument('--parent-selection-type', required=False, default='sss', choices=['sss', 'rws', 'sus', 'random', 'tournament', 'rank'])
    parser.add_argument('--keep-parents', required=False, default=-1, type=int)
    parser.add_argument('--crossover-type', required=False, default='single_point', choices=["single_point", "two_points", "uniform", "scattered"])
    parser.add_argument('--mutation-type', required=False, default='random', choices=["random","swap","scramble","inversion","adaptive"])
    parser.add_argument('--mutation-percent-genes', required=False, default=40, type=int)
    parser.add_argument('--mutation-probability', required=False, default=0.1, type=float)
    return parser


def main():
    parser = get_parser()
    args = vars(parser.parse_args())
    args['people'] = import_people(os.path.join(BASE_PATH, 'testdata', args['test'], 'people.csv'))
    args['constraints'] = import_constraints(os.path.join(BASE_PATH, 'testdata', args['test'], 'constraints.csv'))
    args['constraints'].extend([MaxHoursPerWeekConstraint(person) for person in args['people']])
    args['constraints'].extend([MaxHoursPerDayConstraint(person) for person in args['people']])
    hours_to_schedule = set()
    for constraint in args['constraints']:
        if isinstance(constraint, ScheduledHoursConstraint):
            hours_to_schedule.update(constraint.hours_to_schedule_in_range(args['from'], args['to']))
    args['hours_to_schedule'] = sorted(hours_to_schedule)
    run_ga(args)


if __name__ == "__main__":
    main()
