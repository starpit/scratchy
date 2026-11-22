// SPDX-License-Identifier: Apache-2.0
//! Integers as the program states them: affine in the walk's trip and in the parallel axes.
//!
//! The lift never runs a loop or a grid point by point. A value computed from an induction
//! variable or from `program_id` stays a formula, `c + trip·t + Σ axis[i]·a[i]`, through every
//! integer op that keeps it affine. An address built from it therefore states how it moves: its
//! trip coefficient is how far it steps per walk trip, and each axis coefficient is its stride
//! along a parallel axis (a grid dimension, or a loop whose trips are independent).

/// How many parallel axes one program may open at once: the grid's three and five loops.
pub const AXES: usize = 8;

/// What an affine integer depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sym {
    /// The trip of the program's walk, counting from 0.
    Trip,
    /// The position along parallel axis `i`.
    Axis(usize),
}

/// `c + trip·trip + Σ axis[i]·position(i)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Lin {
    pub c: i64,
    pub trip: i64,
    pub axis: [i64; AXES],
}

/// An integer op whose result is not affine in the symbols it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotAffine;

impl Lin {
    pub const fn constant(c: i64) -> Lin {
        Lin {
            c,
            trip: 0,
            axis: [0; AXES],
        }
    }

    /// The symbol itself.
    pub const fn sym(s: Sym) -> Lin {
        let mut l = Lin::constant(0);
        match s {
            Sym::Trip => l.trip = 1,
            Sym::Axis(i) => l.axis[i] = 1,
        }
        l
    }

    /// Whether the value depends on no symbol.
    const fn symbolic(self) -> bool {
        if self.trip != 0 {
            return true;
        }
        let mut i = 0;
        while i < AXES {
            if self.axis[i] != 0 {
                return true;
            }
            i += 1;
        }
        false
    }

    /// `Some(c)` when the value depends on no symbol.
    pub const fn as_const(self) -> Option<i64> {
        if self.symbolic() { None } else { Some(self.c) }
    }

    fn map(self, k: i64, f: fn(i64, i64) -> i64) -> Lin {
        let mut axis = [0; AXES];
        let mut i = 0;
        while i < AXES {
            axis[i] = f(self.axis[i], k);
            i += 1;
        }
        Lin {
            c: f(self.c, k),
            trip: f(self.trip, k),
            axis,
        }
    }

    pub const fn add(self, o: Lin) -> Lin {
        let mut axis = [0; AXES];
        let mut i = 0;
        while i < AXES {
            axis[i] = self.axis[i] + o.axis[i];
            i += 1;
        }
        Lin {
            c: self.c + o.c,
            trip: self.trip + o.trip,
            axis,
        }
    }

    pub fn scale(self, k: i64) -> Lin {
        self.map(k, |x, k| x * k)
    }

    pub fn minus(self, o: Lin) -> Lin {
        self.add(o.scale(-1))
    }

    /// The product, affine only when one side is a constant.
    pub fn times(self, o: Lin) -> Result<Lin, NotAffine> {
        match (self.as_const(), o.as_const()) {
            (Some(k), _) => Ok(o.scale(k)),
            (_, Some(k)) => Ok(self.scale(k)),
            (None, None) => Err(NotAffine),
        }
    }

    /// Whether every symbolic coefficient is a multiple of `d`.
    const fn symbols_divide(self, d: i64) -> bool {
        if self.trip % d != 0 {
            return false;
        }
        let mut i = 0;
        while i < AXES {
            if self.axis[i] % d != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Floor division by a constant: affine when every symbolic coefficient is a multiple of the
    /// divisor (the symbolic part divides exactly; the constant part floors).
    pub fn div_floor(self, d: i64) -> Result<Lin, NotAffine> {
        if d == 0 || !self.symbols_divide(d) {
            return Err(NotAffine);
        }
        let mut q = self.map(d, |x, d| x / d);
        q.c = self.c.div_euclid(d);
        Ok(q)
    }

    /// The remainder by a constant: a constant when every symbolic coefficient is a multiple of
    /// the divisor.
    pub fn modulo(self, d: i64) -> Result<Lin, NotAffine> {
        if d == 0 || !self.symbols_divide(d) {
            return Err(NotAffine);
        }
        Ok(Lin::constant(self.c.rem_euclid(d)))
    }

    /// `ceil(self / d)` for a positive constant `d`.
    pub fn div_ceil(self, d: i64) -> Result<Lin, NotAffine> {
        self.add(Lin::constant(d - 1)).div_floor(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_address_built_from_the_trip_keeps_its_step() {
        // `page_index_table + i * 32` at trip `i`.
        let a = Lin::constant(5).add(Lin::sym(Sym::Trip).times(Lin::constant(32)).unwrap());
        assert_eq!(a.trip, 32);
        assert_eq!(a.c, 5);
        assert_eq!(a.as_const(), None);
    }

    #[test]
    fn a_product_of_two_symbols_is_not_affine() {
        let t = Lin::sym(Sym::Trip);
        assert_eq!(t.times(Lin::sym(Sym::Axis(0))), Err(NotAffine));
    }

    #[test]
    fn division_is_affine_only_when_the_symbols_divide() {
        let a = Lin::sym(Sym::Trip).scale(64).add(Lin::constant(70));
        let q = a.div_floor(32).unwrap();
        assert_eq!((q.c, q.trip), (2, 2));
        assert_eq!(a.modulo(32), Ok(Lin::constant(6)));
        assert_eq!(Lin::sym(Sym::Trip).div_floor(2), Err(NotAffine));
    }
}
