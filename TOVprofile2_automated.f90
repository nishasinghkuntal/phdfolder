program tov_automation_simple
    ! ------------------------------------------------------------------
    !  Simple, readable TOV solver (physics + style kept from
    !  "TOVprofile 2.f90") with the automation of "TOVprofile.f90":
    !     * loop over several EOS files
    !     * scan central density -> build the mass-radius (M-R) curve
    !     * find M_max and the stable branch
    !     * root-find (bisection) the central pressure that gives each
    !       target mass 1.0, 1.1, ... up to M_max and write its profile
    !
    !  The physics equations, units (G = c = 1, lengths in km) and the
    !  RK4 integrator are exactly those of the simple code; only the EOS
    !  reader was upgraded to read the real crust-joined tables and
    !  crust markers / automation were added.
    ! ------------------------------------------------------------------
    implicit none

    ! ---- precision & sizes -------------------------------------------
    ! Was selected_real_kind(33, 4931) -- quadruple precision, which gcc
    ! emulates in software and which made every run ~7x slower than it
    ! needed to be.  Double precision reproduces the quad results to 12-13
    ! significant figures (checked on GM1/GM2/GM3/NL3wr and APR), far
    ! beyond anything physically meaningful here.
    integer, parameter :: qp = selected_real_kind(15, 307)
    integer, parameter :: nmax = 60000, max_eos = 1024, path_len = 512
    integer, parameter :: max_targets = 200

    ! ---- physical constants (geometrized units, G = c = 1) -----------
    real(qp), parameter :: G   = 1.0_qp
    real(qp), parameter :: c   = 1.0_qp
    real(qp), parameter :: pi  = 3.141592653589793_qp
    ! G*Msun/c^2 in km, from the IAU nominal GM_sun = 1.3271244e20 m^3/s^2.
    ! The older value 1.476 overstated every mass by 0.042% (~0.001 Msun
    ! on M_max).  Radii are unaffected: R is integrated in km directly.
    real(qp), parameter :: Msun = 1.4766250_qp      ! G*Msun/c^2 in km

    ! ---- unit conversions (CGS table -> code units) ------------------
    real(qp), parameter :: multiplier_col1 = 0.742367e-18  ! g/cm^3   -> km^-2
    real(qp), parameter :: multiplier_col2 = 0.8260e-39    ! dyn/cm^2 -> km^-2
    real(qp), parameter :: multiplier_col3 = 1.0e+54_qp    ! #/fm^3   -> code
    real(qp), parameter :: Mnucleon = 1.23977e-57_qp       ! baryon mass (code)

    ! ---- integrator / crust / sanity ---------------------------------
    real(qp), parameter :: h        = 0.01_qp        ! radial step (km)
    ! The 10 m step is fine in the core, but in the outer crust the
    ! pressure scale height falls to metres, so a full step can jump the
    ! pressure straight past zero and the integration stops ~17 decades
    ! in pressure above the real surface (radius came out ~0.6% short).
    ! Fix: refuse a step that would drop P by more than dP_frac, halving
    ! it until it is safe.  Only bites near the surface; in the core the
    ! relative change over 10 m is tiny and the step stays exactly h.
    real(qp), parameter :: dP_frac  = 0.1_qp         ! max |dP|/P per step
    real(qp), parameter :: h_min    = 1.0e-7_qp      ! km, smallest step allowed
    real(qp), parameter :: nb_core  = 0.1324_qp      ! #/fm^3, core-crust
    real(qp), parameter :: nb_drip  = 2.573e-4_qp    ! #/fm^3, neutron drip
    real(qp), parameter :: r_ns_max = 30.0_qp        ! km, sanity cap
    real(qp), parameter :: m_min    = 0.05_qp        ! Msun, ignore tiny models
    real(qp), parameter :: nb_c_min = 0.08_qp        ! #/fm^3, min central density scanned

    ! ---- shared work arrays for one TOV profile (host association) ----
    real(qp), allocatable :: rv(:), nbv(:), rhov(:), Pv(:), mv(:), mbv(:), phiv(:)

    integer :: n_eos_files, ieos
    character(len=path_len) :: eos_files(max_eos)
    ! --curve-only: write just the M-R curve and skip the per-target-mass
    ! profiles.  The (Esym,Lsym) grid needs only M_max, R_1.4 and M_DU,
    ! all of which come off the curve; writing ~14 profiles for each of
    ! 2109 models would cost several GB and most of the runtime.
    logical :: curve_only = .false.

    allocate(rv(nmax), nbv(nmax), rhov(nmax), Pv(nmax), mv(nmax), mbv(nmax), phiv(nmax))

    call parse_command_line(n_eos_files, eos_files)

    open(unit=98, file='MRcurve_all_eos_full.dat', status='replace')
    write(98, '(A)') '# eos_name mass_Msun radius_km rho_c_gcm3 P_c_dyncm2'

    do ieos = 1, n_eos_files
        print *, '----------------------------------------'
        print *, 'EOS [', ieos, '/', n_eos_files, ']: ', trim(eos_files(ieos))
        call process_one_eos(trim(eos_files(ieos)))
    end do
    close(98)

    print *, 'Combined MR file: MRcurve_all_eos_full.dat'

contains

    ! ==================================================================
    !  Command line: EOS files, or the default four core+crust tables
    ! ==================================================================
    subroutine parse_command_line(n_files, files)
        integer, intent(out) :: n_files
        character(len=*), intent(out) :: files(:)
        integer :: argc, iarg
        character(len=path_len) :: arg

        argc = command_argument_count()
        n_files = 0
        do iarg = 1, argc
            call get_command_argument(iarg, arg)
            if (len_trim(arg) == 0) cycle
            if (trim(arg) == '--curve-only') then
                curve_only = .true.
                cycle
            end if
            n_files = n_files + 1
            if (n_files > size(files)) stop 'Too many EOS files on command line.'
            files(n_files) = trim(arg)
        end do

        ! Nothing named on the command line.  This used to fall back to
        ! four hardcoded file names (BSR4.dat and friends), which are not
        ! present in most working directories, so the program announced
        ! four missing files and did nothing.  Look for what is actually
        ! on disk instead.
        if (n_files == 0) call discover_eos_files(n_files, files)

        if (n_files == 0) then
            print *, 'No EoS tables given, and none found in NSCool/EOS/.'
            print *, ''
            print *, 'This program reads CRUST-JOINED tables, the ones'
            print *, 'named <MODEL>_J<Esym>_L<Lsym>_CAT.dat.  Build one first:'
            print *, ''
            print *, '    python3 rmf_lambda_and_eos.py'
            print *, '    python3 automationandformatting-2.py'
            print *, ''
            print *, 'then run this program again, or name a file:'
            print *, ''
            print *, '    ./tovprofile NSCool/EOS/GM1_J33p4_L69p0_CAT.dat'
            stop 1
        end if
    end subroutine parse_command_line

    ! ==================================================================
    !  Find the crust-joined tables that actually exist.
    !
    !  Fortran has no directory listing of its own, so ask the shell for
    !  one.  If the shell call fails for any reason the list simply comes
    !  back empty and the caller prints the usage message; nothing here
    !  can silently pick up the wrong file.
    ! ==================================================================
    subroutine discover_eos_files(n_files, files)
        integer, intent(inout) :: n_files
        character(len=*), intent(inout) :: files(:)
        character(len=path_len) :: dir, line
        character(len=*), parameter :: listfile = '.tov_eos_list.tmp'
        integer :: u, ios

        dir = nscool_dir('EOS')
        if (len_trim(dir) == 0) dir = './'

        ! Crust-joined tables: <MODEL>_J<Esym>_L<Lsym>_CAT.dat
        call execute_command_line('ls -1 ' // trim(dir) // &
             '*_CAT.dat 2>/dev/null | grep -v -e MRcurve_ -e TOVprofile_' // &
             ' > ' // listfile // ' 2>/dev/null')

        open(newunit=u, file=listfile, status='old', iostat=ios)
        if (ios /= 0) return
        do
            read(u, '(A)', iostat=ios) line
            if (ios /= 0) exit
            if (len_trim(line) == 0) cycle
            if (n_files >= size(files)) exit
            n_files = n_files + 1
            files(n_files) = trim(line)
        end do
        close(u, status='delete')

        if (n_files > 0) then
            print *, 'No files named; found ', n_files, ' in ', trim(dir)
        end if
    end subroutine discover_eos_files

    ! ==================================================================
    !  Everything for one EOS: M-R curve, M_max, target-mass profiles
    ! ==================================================================
    ! ==================================================================
    !  Where to put the output.
    !
    !  The M-R curve belongs in NSCool/EOS/ and the stellar profiles in
    !  NSCool/TOV/Profile/, which is exactly where NSCool looks for them.
    !  Writing them there directly removes the copy step that used to sit
    !  between this program and the cooling run.  If NSCool is not
    !  installed beside this program, everything falls back to the
    !  current directory and the code behaves as it always did.
    ! ==================================================================
    function nscool_dir(sub) result(d)
        character(len=*), intent(in) :: sub
        character(len=path_len) :: d
        logical :: there
        inquire(file='NSCool/EOS/APR_EOS_Cat.dat', exist=there)
        if (there) then
            d = 'NSCool/' // trim(sub) // '/'
            return
        end if
        inquire(file='../NSCool/EOS/APR_EOS_Cat.dat', exist=there)
        if (there) then
            d = '../NSCool/' // trim(sub) // '/'
            return
        end if
        inquire(file='../../NSCool/EOS/APR_EOS_Cat.dat', exist=there)
        if (there) then
            d = '../../NSCool/' // trim(sub) // '/'
            return
        end if
        d = ''
    end function nscool_dir

    subroutine process_one_eos(eos_path)
        character(len=*), intent(in) :: eos_path

        real(qp), allocatable :: eos_rho(:), eos_P(:), eos_nb(:)
        real(qp), allocatable :: cM(:), cR(:), cPc(:), cRHOc(:)
        integer :: n_eos, i, count, imax, it, n_targets, mass10, unit_curve
        real(qp) :: Mout, Rout, Pc, target_mass, max_mass
        character(len=path_len) :: eos_name, curve_file, prof_file

        allocate(eos_rho(nmax), eos_P(nmax), eos_nb(nmax))
        call read_eos(eos_path, n_eos, eos_rho, eos_P, eos_nb)
        if (n_eos < 3) then
            print *, 'Skipping EOS (not enough points): ', trim(eos_path)
            return
        end if
        call eos_basename(eos_path, eos_name)

        ! ---- scan central density: one TOV solve per EOS entry --------
        allocate(cM(n_eos), cR(n_eos), cPc(n_eos), cRHOc(n_eos))
        count = 0
        do i = 1, n_eos
            if (eos_nb(i) / multiplier_col3 < nb_c_min) cycle   ! core densities only
            call tov_run(eos_P(i), eos_rho, eos_P, eos_nb, n_eos, Mout, Rout, .false., '', '')
            if (Mout /= Mout) cycle                              ! NaN guard
            if (Mout < m_min) cycle
            if (Rout <= 0.0_qp .or. Rout > r_ns_max) cycle
            count = count + 1
            cPc(count)   = eos_P(i)
            cRHOc(count) = eos_rho(i)
            cM(count)    = Mout
            cR(count)    = Rout
        end do

        if (count < 2) then
            print *, 'Skipping EOS (no valid models): ', trim(eos_path)
            return
        end if

        ! ---- stable branch = up to the first (global) maximum mass ----
        imax = 1
        do i = 2, count
            if (cM(i) > cM(imax)) imax = i
        end do
        max_mass = cM(imax)

        ! ---- write the M-R curve (stable branch) ---------------------
        curve_file = trim(nscool_dir('EOS')) // 'MRcurve_' // trim(eos_name) // '.dat'
        open(newunit=unit_curve, file=trim(curve_file), status='replace')
        write(unit_curve, '(A)') '# mass_Msun radius_km rho_c_gcm3 P_c_dyncm2  (stable branch)'
        do i = 1, imax
            if (cM(i) < m_min) cycle
            write(unit_curve, '(1p4e16.8)') cM(i), cR(i), &
                cRHOc(i) / multiplier_col1, cPc(i) / multiplier_col2
            write(98, '(A,1x,1p4e16.8)') trim(eos_name), cM(i), cR(i), &
                cRHOc(i) / multiplier_col1, cPc(i) / multiplier_col2
        end do
        close(unit_curve)

        ! ---- target masses 1.0, 1.1, ... up to M_max -----------------
        if (curve_only) then
            n_targets = 0
        else if (max_mass < 1.0_qp) then
            n_targets = 0
        else
            n_targets = int((max_mass - 1.0_qp) * 10.0_qp + 1.0e-9_qp) + 1
        end if
        if (n_targets > max_targets) n_targets = max_targets

        do it = 1, n_targets
            target_mass = 1.0_qp + 0.1_qp * real(it - 1, qp)
            if (target_mass > max_mass + 1.0e-6_qp) exit

            call find_pc_for_mass(target_mass, eos_rho, eos_P, eos_nb, n_eos, &
                cM, cPc, imax, Pc, Mout)

            if (Mout < 0.5_qp .or. Mout /= Mout) cycle
            if (abs(Mout - target_mass) > 0.05_qp .and. target_mass < max_mass - 0.05_qp) cycle

            mass10 = nint(target_mass * 10.0_qp)
            write(prof_file, '(A,"_targetM",I0,".dat")') &
                trim(nscool_dir('TOV/Profile')) // 'TOVprofile_' // trim(eos_name), mass10
            call tov_run(Pc, eos_rho, eos_P, eos_nb, n_eos, Mout, Rout, .true., &
                trim(eos_path), trim(adjustl(prof_file)))
        end do

        ! ---- always write the maximum-mass profile -------------------
        if (max_mass >= 1.0_qp .and. .not. curve_only) then
            prof_file = trim(nscool_dir('TOV/Profile')) // 'TOVprofile_' // trim(eos_name) // '_Mmax.dat'
            call tov_run(cPc(imax), eos_rho, eos_P, eos_nb, n_eos, Mout, Rout, .true., &
                trim(eos_path), trim(prof_file))
        end if

        print *, 'EOS: ', trim(eos_name), '  Mmax =', max_mass, ' Msun,  R(Mmax) =', cR(imax), ' km'
        print *, '  MR curve: ', trim(curve_file)
    end subroutine process_one_eos

    ! ==================================================================
    !  One TOV integration (RK4, fixed step) — identical physics to the
    !  simple code.  Returns M (Msun) and R (km); optionally writes the
    !  full radial profile to a file.
    ! ==================================================================
    subroutine tov_run(Pc, eos_rho, eos_P, eos_nb, n_eos, Mout, Rout, do_write, eos_title, out_file)
        real(qp), intent(in) :: Pc
        real(qp), intent(in) :: eos_rho(:), eos_P(:), eos_nb(:)
        integer, intent(in) :: n_eos
        real(qp), intent(out) :: Mout, Rout
        logical, intent(in) :: do_write
        character(len=*), intent(in) :: eos_title, out_file

        real(qp) :: r, P, m, mb, rho, nb, phi, phi_surface, dp, P_stop, hs
        real(qp) :: rho1, rho2, rho3, rho4, nb1, nb2, nb3, nb4
        real(qp) :: k1, k2, k3, k4, l1, l2, l3, l4, n1, n2, n3, n4
        integer :: step, num_steps, icore, idrip, j, unit_out

        P_stop = eos_P(1)          ! lowest tabulated pressure (ascending table)

        ! ---- central boundary conditions -----------------------------
        r  = 1.0e-6_qp
        m  = 0.0_qp
        mb = 0.0_qp
        phi = 0.0_qp
        P  = Pc
        call interpolate(eos_P, eos_rho, n_eos, P, rho)
        call interpolate(eos_P, eos_nb,  n_eos, P, nb)
        step = 0; icore = 0; idrip = 0

        ! ---- RK4 outward until pressure reaches the table bottom ------
        do while (P >= P_stop .and. step < nmax - 1)
            step = step + 1
            call interpolate(eos_P, eos_rho, n_eos, P, rho)
            call interpolate(eos_P, eos_nb,  n_eos, P, nb)

            ! same RK4 as before; the only change is that the step hs
            ! shrinks if a full step would take too big a bite out of P
            hs = h
            do
                ! The density must be looked up again at every RK4 stage.
                ! Holding rho at its start-of-step value turns the mass
                ! integral into a left-endpoint rule, which on a falling
                ! rho(r) overestimates the mass by ~0.7%.
                call interpolate(eos_P, eos_rho, n_eos, P, rho1)
                call interpolate(eos_P, eos_nb,  n_eos, P, nb1)
                k1 = hs * pressure_eq(r,            m,             P,            rho1)
                l1 = hs * mass_eq(r, rho1)
                n1 = hs * mb_eq(r, m, nb1)

                call interpolate(eos_P, eos_rho, n_eos, P + 0.5_qp*k1, rho2)
                call interpolate(eos_P, eos_nb,  n_eos, P + 0.5_qp*k1, nb2)
                k2 = hs * pressure_eq(r + 0.5_qp*hs, m + 0.5_qp*l1, P + 0.5_qp*k1, rho2)
                l2 = hs * mass_eq(r + 0.5_qp*hs, rho2)
                n2 = hs * mb_eq(r + 0.5_qp*hs, m + 0.5_qp*l1, nb2)

                call interpolate(eos_P, eos_rho, n_eos, P + 0.5_qp*k2, rho3)
                call interpolate(eos_P, eos_nb,  n_eos, P + 0.5_qp*k2, nb3)
                k3 = hs * pressure_eq(r + 0.5_qp*hs, m + 0.5_qp*l2, P + 0.5_qp*k2, rho3)
                l3 = hs * mass_eq(r + 0.5_qp*hs, rho3)
                n3 = hs * mb_eq(r + 0.5_qp*hs, m + 0.5_qp*l2, nb3)

                call interpolate(eos_P, eos_rho, n_eos, P + k3, rho4)
                call interpolate(eos_P, eos_nb,  n_eos, P + k3, nb4)
                k4 = hs * pressure_eq(r + hs, m + l3, P + k3, rho4)
                l4 = hs * mass_eq(r + hs, rho4)
                n4 = hs * mb_eq(r + hs, m + l3, nb4)

                dp = (k1 + 2.0_qp*k2 + 2.0_qp*k3 + k4) / 6.0_qp
                if (abs(dp) <= dP_frac * P) exit
                if (hs <= h_min) exit
                hs = 0.5_qp * hs
            end do

            r  = r + hs
            phi = phi - dp / (rho + P)
            P  = P  + dp
            m  = m  + (l1 + 2.0_qp*l2 + 2.0_qp*l3 + l4) / 6.0_qp
            mb = mb + (n1 + 2.0_qp*n2 + 2.0_qp*n3 + n4) / 6.0_qp

            ! the RK4 step can overshoot below the surface: drop that point
            if (P <= 0.0_qp) then
                step = step - 1
                exit
            end if

            ! rho and nb belong with the P we just reached, not with the
            ! one we started the step from
            call interpolate(eos_P, eos_rho, n_eos, P, rho)
            call interpolate(eos_P, eos_nb,  n_eos, P, nb)

            ! crust markers (row index = step-1, matching the output rows)
            if (icore == 0 .and. nb / multiplier_col3 <= nb_core) icore = step - 1
            if (idrip == 0 .and. nb / multiplier_col3 <  nb_drip) idrip = step - 1

            rv(step)  = r
            nbv(step) = nb
            rhov(step)= rho
            Pv(step)  = P
            mv(step)  = m
            mbv(step) = mb
            phiv(step)= phi
        end do
        num_steps = step

        if (num_steps < 1) then
            Mout = 0.0_qp; Rout = 0.0_qp
            return
        end if

        Mout = mv(num_steps) / Msun
        Rout = rv(num_steps)

        ! ---- rescale phi so it matches the exterior Schwarzschild metric
        if (r > 2.0_qp * m) then
            phi_surface = 0.5_qp * log(1.0_qp - 2.0_qp * m / r)
            do j = 1, num_steps
                phiv(j) = phiv(j) - phiv(num_steps) + phi_surface
            end do
        end if

        if (.not. do_write) return

        ! ---- write profile (NSCool-style numeric header + columns) ----
        open(newunit=unit_out, file=out_file, status='replace')
        write(unit_out, '(4i8)') 6, num_steps - 1, icore, idrip
        write(unit_out, *)
        write(unit_out, *) '    EOS file :  ', trim(eos_title)
        write(unit_out, *)
        write(unit_out, '(A6,A15,A15,A15,A15,A18,A15,A18)') &
            'step', 'radius ', 'baryon#  ', 'density   ', '  pressure  ', 'encl. mass   ', 'phi   ', &
            'encl. bar. mass '
        write(unit_out, '(A6,A15,A15,A15,A15,A18,A15,A18)') &
            '   ', '  (m)  ', '(#/fm3)  ', '(g/cm3)   ', '  (dyn/cm2) ', ' (sol. mass)  ', '     ', &
            ' (sol. mass)   '
        write(unit_out, *)
        do j = 1, num_steps
            write(unit_out, '(i6,0pf15.6,1p1e15.6,1e15.6,1e15.5,1e18.9,1e15.6,1e18.9)') &
                j - 1, rv(j) * 1000.0_qp, nbv(j) / multiplier_col3, rhov(j) / multiplier_col1, &
                Pv(j) / multiplier_col2, mv(j) / Msun, phiv(j), mbv(j) / Msun
        end do
        close(unit_out)
    end subroutine tov_run

    ! ==================================================================
    !  Bisection: central pressure Pc such that M(Pc) = target_mass,
    !  on the stable branch where M(Pc) is monotonically increasing.
    ! ==================================================================
    subroutine find_pc_for_mass(target, eos_rho, eos_P, eos_nb, n_eos, cM, cPc, imax, Pc_out, M_out)
        real(qp), intent(in) :: target
        real(qp), intent(in) :: eos_rho(:), eos_P(:), eos_nb(:), cM(:), cPc(:)
        integer, intent(in) :: n_eos, imax
        real(qp), intent(out) :: Pc_out, M_out

        real(qp) :: plo, phi_hi, mlo, mhi, pmid, mmid, Rout
        integer :: k, iter
        logical :: bracketed

        bracketed = .false.
        do k = 1, imax - 1
            if (cM(k) <= target .and. cM(k+1) >= target) then
                plo = cPc(k); phi_hi = cPc(k+1)
                bracketed = .true.
                exit
            end if
        end do

        if (.not. bracketed) then
            Pc_out = cPc(imax)
            call tov_run(Pc_out, eos_rho, eos_P, eos_nb, n_eos, M_out, Rout, .false., '', '')
            return
        end if

        call tov_run(plo,    eos_rho, eos_P, eos_nb, n_eos, mlo, Rout, .false., '', '')
        call tov_run(phi_hi, eos_rho, eos_P, eos_nb, n_eos, mhi, Rout, .false., '', '')

        pmid = 0.5_qp * (plo + phi_hi)
        mmid = mlo
        do iter = 1, 60
            pmid = 0.5_qp * (plo + phi_hi)
            call tov_run(pmid, eos_rho, eos_P, eos_nb, n_eos, mmid, Rout, .false., '', '')
            if (abs(mmid - target) < 1.0e-6_qp) exit
            if ((mlo - target) * (mmid - target) <= 0.0_qp) then
                phi_hi = pmid; mhi = mmid
            else
                plo = pmid; mlo = mmid
            end if
        end do

        Pc_out = pmid
        M_out  = mmid
    end subroutine find_pc_for_mass

    ! ==================================================================
    !  TOV right-hand sides — kept exactly as in the simple code
    ! ==================================================================
    function pressure_eq(r, m, P, rho)     ! dP/dr  (TOV equation)
        real(qp), intent(in) :: r, m, P, rho
        real(qp) :: pressure_eq
        pressure_eq = - (rho + P) * (m + 4.0_qp * pi * r**3 * P) / (r * (r - 2.0_qp * m))
    end function pressure_eq

    function mass_eq(r, rho)               ! dm/dr
        real(qp), intent(in) :: r, rho
        real(qp) :: mass_eq
        mass_eq = 4.0_qp * pi * r**2 * rho
    end function mass_eq

    function mb_eq(r, m, nb)               ! d(baryon mass)/dr  (proper volume)
        real(qp), intent(in) :: r, m, nb
        real(qp) :: mb_eq
        mb_eq = (4.0_qp * pi * r**2 * nb * Mnucleon) / sqrt(1.0_qp - (2.0_qp * m / r))
    end function mb_eq

    ! ==================================================================
    !  Linear interpolation (table must be ascending in x = pressure)
    ! ==================================================================
    subroutine interpolate(xarr, yarr, n, xv, yv)
        integer, intent(in) :: n
        real(qp), intent(in) :: xarr(:), yarr(:), xv
        real(qp), intent(out) :: yv
        integer :: i

        if (xv <= xarr(1)) then
            yv = yarr(1); return
        end if
        if (xv >= xarr(n)) then
            yv = yarr(n); return
        end if
        do i = 1, n - 1
            if (xv >= xarr(i) .and. xv <= xarr(i+1)) then
                yv = yarr(i) + (xv - xarr(i)) * (yarr(i+1) - yarr(i)) / (xarr(i+1) - xarr(i))
                return
            end if
        end do
        yv = yarr(n)
    end subroutine interpolate

    ! ==================================================================
    !  Read a real crust-joined <MODEL>_*_CAT.dat table:
    !    * skip 6 header lines
    !    * columns 1,2,3 = rho(g/cm3), P(dyn/cm2), nbar(#/fm3)
    !    * drop trailing "\ ..." comments, stop at "CRUST: from"
    !    * convert to code units and REVERSE to ascending order
    ! ==================================================================
    subroutine read_eos(file_name, n_eos, eos_rho, eos_P, eos_nb)
        character(len=*), intent(in) :: file_name
        integer, intent(out) :: n_eos
        real(qp), intent(out) :: eos_rho(:), eos_P(:), eos_nb(:)

        real(qp) :: rho_v, P_v, nb_v, tr, tp, tn
        integer :: i, ios, bs, k, nn
        character(len=512) :: line

        open(unit=20, file=file_name, status='old', iostat=ios)
        if (ios /= 0) then
            n_eos = 0; return
        end if

        do i = 1, 6                        ! skip header
            read(20, *, iostat=ios)
            if (ios /= 0) then
                close(20); n_eos = 0; return
            end if
        end do

        nn = 0
        do
            read(20, '(A)', iostat=ios) line
            if (ios /= 0) exit
            if (len_trim(line) == 0) cycle
            if (index(line, 'CRUST: from') > 0) exit
            bs = index(line, char(92))     ! backslash -> trailing comment
            if (bs > 0) line = line(1:bs-1)
            read(line, *, iostat=ios) rho_v, P_v, nb_v
            if (ios /= 0) cycle
            nn = nn + 1
            if (nn > nmax) exit
            eos_rho(nn) = rho_v * multiplier_col1
            eos_P(nn)   = P_v   * multiplier_col2
            eos_nb(nn)  = nb_v  * multiplier_col3
        end do
        close(20)

        ! reverse: the file is highest-density first, interpolation needs ascending
        do k = 1, nn / 2
            tr = eos_rho(k); eos_rho(k) = eos_rho(nn-k+1); eos_rho(nn-k+1) = tr
            tp = eos_P(k);   eos_P(k)   = eos_P(nn-k+1);   eos_P(nn-k+1)   = tp
            tn = eos_nb(k);  eos_nb(k)  = eos_nb(nn-k+1);  eos_nb(nn-k+1)  = tn
        end do
        n_eos = nn
    end subroutine read_eos

    ! ==================================================================
    !  Strip directory and .dat extension from an EOS path
    ! ==================================================================
    subroutine eos_basename(path, base)
        character(len=*), intent(in) :: path
        character(len=*), intent(out) :: base
        integer :: i, p, dot

        p = 0
        do i = len_trim(path), 1, -1
            if (path(i:i) == '/' .or. path(i:i) == char(92)) then
                p = i; exit
            end if
        end do
        if (p == 0) then
            base = adjustl(trim(path))
        else
            base = adjustl(path(p+1:len_trim(path)))
        end if
        dot = 0
        do i = len_trim(base), 1, -1
            if (base(i:i) == '.') then
                dot = i; exit
            end if
        end do
        if (dot > 1) base = base(1:dot-1)
    end subroutine eos_basename

end program tov_automation_simple
