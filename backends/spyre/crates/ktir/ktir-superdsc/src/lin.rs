// SPDX-License-Identifier: Apache-2.0
//! Integers as the program states them: affine in the walk's trip and the launch grid's indices.
//!
//! The lift never runs a loop or a grid point by point. A value computed from `scf.for`'s induction
//! variable or from `program_id` stays a formula, `c + trip·t + Σ pid[d]·p[d]`, through every
//! integer op that keeps it affine. An address built from it therefore states how it moves: its
//! trip coefficient is how far it steps per trip, and each grid coefficient is the stride of a
//! parallel axis.

/// What an affine integer depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sym {
    /// The trip of the program's walk, counting from 0.
    Trip,
    /// `program_id(axis)`.
    Pid(usize),
}

/// `c + trip·trip + Σ pid[d]·grid_index(d)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Lin {
    pub c: i64,
    pub trip: i64,
    pub pid: [i64; 3],
}

/// An integer op whose result is not affine in the symbols it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotAffine;

impl Lin {
    pub const fn constant(c: i64) -> Lin {
        Lin {
            c,
            trip: 0,
            pid: [0; 3],
        }
    }

    /// The symbol itself.
    pub const fn sym(s: Sym) -> Lin {
        let mut l = Lin::constant(0);
        match s {
            Sym::Trip => l.trip = 1,
            Sym::Pid(d) => l.pid[d] = 1,
        }
        l
    }

    /// `Some(c)` when the value depends on no symbol.
    pub const fn as_const(self) -> Option<i64> {
        if self.trip == 0 && self.pid[0] == 0 && self.pid[1] == 0 && self.pid[2] == 0 {
            Some(self.c)
        } else {
            None
        }
    }

    pub const fn add(self, o: Lin) -> Lin {
        Lin {
            c: self.c + o.c,
            trip: self.trip + o.trip,
            pid: [
                self.pid[0] + o.pid[0],
                self.pid[1] + o.pid[1],
                self.pid[2] + o.pid[2],
            ],
        }
    }

    pub const fn scale(self, k: i64) -> Lin {
        Lin {
            c: self.c * k,
            trip: self.trip * k,
            pid: [self.pid[0] * k, self.pid[1] * k, self.pid[2] * k],
        }
    }

    pub const fn sub(self, o: Lin) -> Lin {
        self.add(o.scale(-1))
    }

    /// The product, affine only when one side is a constant.
    pub const fn mul(self, o: Lin) -> Result<Lin, NotAffine> {
        match (self.as_const(), o.as_const()) {
            (Some(k), _) => Ok(o.scale(k)),
            (_, Some(k)) => Ok(self.scale(k)),
            (None, None) => Err(NotAffine),
        }
    }

    /// Floor division by a constant: affine when every coefficient is a multiple of the divisor
    /// (the symbolic part divides exactly; the constant part floors).
    pub const fn div_floor(self, d: i64) -> Result<Lin, NotAffine> {
        if d == 0 {
            return Err(NotAffine);
        }
        let sym_divides = self.trip % d == 0
            && self.pid[0] % d == 0
            && self.pid[1] % d == 0
            && self.pid[2] % d == 0;
        if !sym_divides {
            return Err(NotAffine);
        }
        Ok(Lin {
            c: self.c.div_euclid(d),
            trip: self.trip / d,
            pid: [self.pid[0] / d, self.pid[1] / d, self.pid[2] / d],
        })
    }

    /// The remainder by a constant: affine (a constant) when every coefficient is a multiple of
    /// the divisor.
    pub const fn rem(self, d: i64) -> Result<Lin, NotAffine> {
        match self.div_floor(d) {
            Ok(_) => Ok(Lin::constant(self.c.rem_euclid(d))),
            Err(e) => Err(e),
        }
    }

    /// `ceil(self / d)` for a positive constant `d`.
    pub const fn div_ceil(self, d: i64) -> Result<Lin, NotAffine> {
        match self.add(Lin::constant(d - 1)).div_floor(d) {
            Ok(l) => Ok(l),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_address_built_from_the_trip_keeps_its_step() {
        // `page_index_table + i * 32` at trip `i`.
        let a = Lin::constant(5).add(Lin::sym(Sym::Trip).mul(Lin::constant(32)).unwrap());
        assert_eq!(a, Lin { c: 5, trip: 32, pid: [0; 3] });
        assert_eq!(a.as_const(), None);
    }

    #[test]
    fn a_product_of_two_symbols_is_not_affine() {
        let t = Lin::sym(Sym::Trip);
        assert_eq!(t.mul(Lin::sym(Sym::Pid(0))), Err(NotAffine));
    }

    #[test]
    fn division_is_affine_only_when_the_symbols_divide() {
        let a = Lin::sym(Sym::Trip).scale(64).add(Lin::constant(70));
        assert_eq!(a.div_floor(32), Ok(Lin { c: 2, trip: 2, pid: [0; 3] }));
        assert_eq!(a.rem(32), Ok(Lin::constant(6)));
        assert_eq!(Lin::sym(Sym::Trip).div_floor(2), Err(NotAffine));
    }
}
